//
// Reference: https://vulkan-tutorial.com.
//
const std = @import("std");
const glfw = @import("glfw.zig");
const vk = @import("vulkan");
const builtin = @import("builtin");
const shaders = @import("shaders.zig");

const print = std.debug.print;

const Allocator = std.mem.Allocator;

pub const DEBUG = (builtin.mode == .Debug);

const validation_layers: [1][]const u8 = .{
    "VK_LAYER_KHRONOS_validation"
};

const required_device_extensions: [1][]const u8 = .{
    vk.extensions.khr_swapchain.name
};

const DebugSeverityFlag = enum(u32) {
    VERBOSE = 1,
    INFO = 16,
    WARNING = 256,
    ERROR = 4096,
};

const DebugTypeFlag = enum(u32) {
    GENERAL = 1,
    VALIDATION = 2,
    PERFORMANCE = 4,
    DEVICE_ADDR = 8,
};

const QueueFamilies = struct {
    graphics_family: ?u32 = null,
    presentation_family: ?u32 = null,

    pub fn complete(self: *const QueueFamilies) bool {
        return self.graphics_family != null and self.presentation_family != null;
    }

    pub fn sameQueue(self: *const QueueFamilies) bool {
        return self.graphics_family == self.presentation_family;
    }

    pub fn asSlice(self: *const QueueFamilies) ?[2]u32 {
        if (!self.complete()) {
            return null;
        }

        return [_]u32{self.graphics_family.?, self.presentation_family.?};
    }
};

const FrameCommandBuffer = struct {
    command_buffer: vk.CommandBuffer,
    image_available_semaphore: vk.Semaphore,
    render_finished_semaphore: vk.Semaphore,
    in_flight_fence: vk.Fence
};

fn makeLayerName(name: []const u8) [vk.MAX_EXTENSION_NAME_SIZE]u8 {
    var result: [vk.MAX_EXTENSION_NAME_SIZE]u8 = undefined;
    @memcpy(result[0..name.len], name);
    result[name.len] = 0;
    @memset(result[name.len+1..], 0);
    return result;
}

fn debugCallback(
    message_severity: vk.DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk.DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
    _: ?*anyopaque
) callconv(vk.vulkan_call_conv) vk.Bool32 {
    if (message_severity.verbose_bit_ext) {
        return vk.FALSE;
    }

    const severity = switch (message_severity.toInt()) {
        @intFromEnum(DebugSeverityFlag.VERBOSE) => "VERBOSE",
        @intFromEnum(DebugSeverityFlag.INFO) => "INFO",
        @intFromEnum(DebugSeverityFlag.WARNING) => "WARNING",
        @intFromEnum(DebugSeverityFlag.ERROR) => "ERROR",
        else => unreachable,
    };

    const msg_type = switch (message_type.toInt()) {
        @intFromEnum(DebugTypeFlag.GENERAL) => "GENERAL",
        @intFromEnum(DebugTypeFlag.VALIDATION) => "VALIDATION",
        @intFromEnum(DebugTypeFlag.PERFORMANCE) => "PERFORMANCE",
        @intFromEnum(DebugTypeFlag.DEVICE_ADDR) => "DEVICE ADDR",
        else => unreachable,
    };

    print(
        "[{d}][{s}][{s}] {?s}\n", 
        .{std.time.microTimestamp(), severity, msg_type, p_callback_data.?.p_message}
    );

    return vk.FALSE;
}

fn initDebugCreateInfo() vk.DebugUtilsMessengerCreateInfoEXT {
    return .{
        .message_severity = .{
            .error_bit_ext = true,
            .info_bit_ext = true,
            .verbose_bit_ext = true,
            .warning_bit_ext = true,
        },
        .message_type = .{
            .general_bit_ext = true,
            .validation_bit_ext = true,
            .performance_bit_ext = true,
        },
        .pfn_user_callback = &debugCallback,
        .p_user_data = null
    };
}

pub const GraphicalContext = struct {
    allocator: Allocator,

    vkb: vk.BaseWrapper,

    instance: vk.InstanceProxy,
    physical_device: vk.PhysicalDevice,
    device: vk.DeviceProxy,
    surface: vk.SurfaceKHR,

    graphics_queue: vk.Queue,
    present_queue: vk.Queue,

    swapchain: vk.SwapchainKHR,
    swapchain_format: vk.SurfaceFormatKHR,
    swapchain_extent: vk.Extent2D,

    swapchain_framebuffers: std.ArrayList(vk.Framebuffer),
    command_pool: vk.CommandPool,

    current_frame: u32 = 0,
    max_frames_in_flight: u32 = 3,
    frame_command_buffers: std.ArrayList(FrameCommandBuffer),

    // command_buffer: vk.CommandBuffer,
    //
    // image_available_semaphore: vk.Semaphore,
    // render_finished_semaphore: vk.Semaphore,
    // in_flight_fence: vk.Fence,

    images: []vk.Image,
    images_view: std.ArrayList(vk.ImageView),

    render_pass: vk.RenderPass,
    pipeline_layout: vk.PipelineLayout,
    graphics_pipeline: vk.Pipeline,

    debug_messenger_ext: ?vk.DebugUtilsMessengerEXT,

    pub fn init(allocator: Allocator, window: *glfw.Window) !GraphicalContext {
        var self: GraphicalContext = undefined;

        self.allocator = allocator;
        self.vkb = vk.BaseWrapper.load(glfw.getInstanceProcAddress);

        try self.initInstance();
        try self.setupDebugMessenger();
        try self.initSurface(window);
        try self.initDevice();
        try self.initSwapchain(window);
        try self.initRenderPass();
        try self.initGraphicsPipeline();
        try self.initFramebuffer();
        try self.initCommandPool();
        try self.initCommandBuffer();

        return self;
    }

    fn initInstance(self: *GraphicalContext) !void {
        const enable_validation_layer = comptime DEBUG;
        if (enable_validation_layer and !try self.checkValidationLayerSupport()) {
            return error.ValidationLayerNotFound;
        }

        var extensions_list = std.ArrayList([*:0]const u8).init(self.allocator);
        defer extensions_list.deinit(); 
        try getRequiredExtensions(&extensions_list);

        const instance = try self.vkb.createInstance(&.{
            .enabled_extension_count = @intCast(extensions_list.items.len),
            .pp_enabled_extension_names = extensions_list.items.ptr,
            .enabled_layer_count = if (DEBUG) validation_layers.len else 0,
            .pp_enabled_layer_names = if (DEBUG) @ptrCast(&validation_layers) else null,
            .p_next = &initDebugCreateInfo(),
            .p_application_info = &.{
                .p_application_name = "Hello",
                .application_version = @bitCast(vk.makeApiVersion(1, 0, 0, 0)),
                .p_engine_name = "hehe",
                .engine_version = @bitCast(vk.makeApiVersion(1, 0, 0, 0)),
                .api_version = @bitCast(vk.API_VERSION_1_3)
            }
        }, null);

        const vki = try self.allocator.create(vk.InstanceWrapper);
        errdefer self.allocator.destroy(vki);

        vki.* = vk.InstanceWrapper.load(instance, self.vkb.dispatch.vkGetInstanceProcAddr.?);
        self.instance = vk.InstanceProxy.init(instance, vki);
        errdefer self.instance.destroyInstance(null);
    }

    fn initSurface(self: *GraphicalContext, window: *glfw.Window) !void {
        var surface: vk.SurfaceKHR = undefined;
        if (glfw.createWindowSurface(self.instance.handle, window, null, &surface) != .success) {
            return error.ErrorCreatingSurface;
        }

        self.surface = surface;
    }

    fn initDevice(self: *GraphicalContext) !void {
        const physical_devices = try self.instance.enumeratePhysicalDevicesAlloc(self.allocator);
        defer self.allocator.free(physical_devices);

        const suitable_device: vk.PhysicalDevice = try self.getSuitableDevice() 
        orelse return error.NoSuitableDeviceFound;

        const queue_family = try self.getQueueFamilies(suitable_device);
        if (!queue_family.complete()) {
            return error.NoCompleteQueueFamilyFound;
        }

        const queue_count: u32 = if (queue_family.sameQueue()) 1 else 2;

        const priority = [_]f32{1};
        const device = try self.instance.createDevice(suitable_device, &.{
            .enabled_extension_count = required_device_extensions.len,
            .pp_enabled_extension_names = @ptrCast(&required_device_extensions),
            .queue_create_info_count = queue_count,
            .p_queue_create_infos = &[_]vk.DeviceQueueCreateInfo{
                .{
                    .queue_family_index = @intCast(queue_family.graphics_family.?),
                    .queue_count = 1,
                    .p_queue_priorities = &priority
                },
                .{
                    .queue_family_index = @intCast(queue_family.presentation_family.?),
                    .queue_count = 1,
                    .p_queue_priorities = &priority
                }
            },
        }, null);

        const vkd = try self.allocator.create(vk.DeviceWrapper);
        errdefer self.allocator.destroy(vkd);

        vkd.* = vk.DeviceWrapper.load(device, self.instance.wrapper.dispatch.vkGetDeviceProcAddr.?);
        self.device = vk.DeviceProxy.init(device, vkd);
        errdefer self.device.destroyDevice(null);
        self.physical_device = suitable_device;

        self.graphics_queue = self.device.getDeviceQueue(queue_family.graphics_family.?, 0);
        self.present_queue = self.device.getDeviceQueue(queue_family.presentation_family.?, 0);
    }

    fn initSwapchain(self: *GraphicalContext, window: *glfw.Window) !void {
        const formats = try self.instance.getPhysicalDeviceSurfaceFormatsAllocKHR(self.physical_device, self.surface, self.allocator);
        defer self.allocator.free(formats);
        
        var cur_format: ?vk.SurfaceFormatKHR = null;
        for (formats) |format| {
            const has_preferred_format = format.format == .b8g8r8_srgb;
            const has_preferred_color_space = format.color_space == .srgb_nonlinear_khr;

            if (has_preferred_format and has_preferred_color_space) {
                cur_format = format;
            }
        }

        if (cur_format == null) {
            cur_format = formats[0];
        }

        const present_modes = try self.instance.getPhysicalDeviceSurfacePresentModesAllocKHR(self.physical_device, self.surface, self.allocator);
        defer self.allocator.free(present_modes);

        var cur_present_mode: ?vk.PresentModeKHR = null;
        for (present_modes) |present_mode| {
            const has_preferred_mode = present_mode == .mailbox_khr;
            if (has_preferred_mode) {
                cur_present_mode = present_mode;
            }
        }

        if (cur_present_mode == null) {
            cur_present_mode = .immediate_khr;
        }

        const capabilities = try self.instance.getPhysicalDeviceSurfaceCapabilitiesKHR(self.physical_device, self.surface);

        var cur_extent: vk.Extent2D = undefined;
        if (capabilities.current_extent.width != std.math.maxInt(u32)) {
            cur_extent = capabilities.current_extent;
        } else {
            var width: u32 = undefined;
            var height: u32 = undefined;
            glfw.getFramebufferSize(window, &width, &height);
            
            width = std.math.clamp(width, capabilities.min_image_extent.width, capabilities.max_image_extent.width);
            height = std.math.clamp(width, capabilities.min_image_extent.height, capabilities.max_image_extent.height);

            cur_extent = .{
                .width = width,
                .height = height,
            };
        }

        var image_count: u32 = capabilities.min_image_count + 1;
        if (capabilities.max_image_count > 0) {
            image_count = @max(image_count, capabilities.max_image_count);
        }

        const queue_family = try self.getQueueFamilies(self.physical_device);
        const queue_slice = queue_family.asSlice().?;
        
        self.swapchain = try self.device.createSwapchainKHR(&.{
            .surface = self.surface,
            .min_image_count = image_count,
            .image_format = cur_format.?.format,
            .image_color_space = cur_format.?.color_space,
            .image_extent = cur_extent,
            .image_array_layers = 1,
            .image_usage = .{ .color_attachment_bit = true, .transfer_dst_bit = true },
            .image_sharing_mode = if (queue_family.sameQueue()) .exclusive else .concurrent,
            .queue_family_index_count = if (queue_family.sameQueue()) 0 else 2,
            .p_queue_family_indices = if (queue_family.sameQueue()) null else &queue_slice,
            .pre_transform = capabilities.current_transform,
            .composite_alpha = .{ .opaque_bit_khr = true },
            .present_mode = cur_present_mode.?,
            .clipped = vk.TRUE
        }, null);
        errdefer self.device.destroySwapchainKHR(self.swapchain, null);

        self.swapchain_format = cur_format.?;
        self.swapchain_extent = cur_extent;

        self.images = try self.device.getSwapchainImagesAllocKHR(self.swapchain, self.allocator);
        errdefer self.allocator.free(self.images);

        self.images_view = std.ArrayList(vk.ImageView).init(self.allocator);
        for (self.images) |image| {
            const image_view = try self.device.createImageView(&.{
                .image = image,
                .view_type = .@"2d",
                .format = self.swapchain_format.format,
                .components = .{
                    .r = .identity,
                    .g = .identity,
                    .b = .identity,
                    .a = .identity,
                },
                .subresource_range = .{
                    .aspect_mask = .{ .color_bit = true },
                    .base_mip_level = 0,
                    .level_count = 1,
                    .base_array_layer = 0,
                    .layer_count = 1,
                }
            }, null);
            errdefer self.device.destroyImageView(image_view, null);

            try self.images_view.append(image_view);
        }
    }

    fn initRenderPass(self: *GraphicalContext) !void {
        const color_attachment = vk.AttachmentDescription{
            .format = self.swapchain_format.format,
            .samples = .{ .@"1_bit" = true },
            .load_op = .clear,
            .store_op = .store,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .undefined,
            .final_layout = .present_src_khr,
        };

        const color_attachment_ref = vk.AttachmentReference{
            .attachment = 0,
            .layout = .color_attachment_optimal
        };

        const subpass = vk.SubpassDescription{
            .pipeline_bind_point = .graphics,
            .color_attachment_count = 1,
            .p_color_attachments = @ptrCast(&color_attachment_ref)
        };

        const dependency = vk.SubpassDependency{
            .src_subpass = vk.SUBPASS_EXTERNAL,
            .dst_subpass = 0,
            .src_stage_mask = .{.color_attachment_output_bit = true},
            .src_access_mask = .{},
            .dst_stage_mask = .{.color_attachment_output_bit = true},
            .dst_access_mask = .{.color_attachment_write_bit = true},
        };

        self.render_pass = try self.device.createRenderPass(&.{
            .attachment_count = 1,
            .p_attachments = @ptrCast(&color_attachment),
            .subpass_count = 1,
            .p_subpasses = @ptrCast(&subpass),
            .dependency_count = 1,
            .p_dependencies = @ptrCast(&dependency)
        }, null);
    }

    fn initGraphicsPipeline(self: *GraphicalContext) !void {
        const vert_spirv = try shaders.compileShader(self.allocator, "triangle_vert.glsl", .vertex);
        defer self.allocator.free(vert_spirv);

        const frag_spirv = try shaders.compileShader(self.allocator, "triangle_frag.glsl", .fragment);
        defer self.allocator.free(frag_spirv);

        const vert_shader_module = try self.createShaderModule(vert_spirv);
        defer self.device.destroyShaderModule(vert_shader_module, null);

        const frag_shader_module = try self.createShaderModule(frag_spirv);
        defer self.device.destroyShaderModule(frag_shader_module, null);

        const shader_stages = [_]vk.PipelineShaderStageCreateInfo{
            .{
                .stage = .{ .vertex_bit = true },
                .p_name = "main",
                .module = vert_shader_module
            },
            .{
                .stage = .{ .fragment_bit = true },
                .p_name = "main",
                .module = frag_shader_module
            },
        };


        const dynamic_states = [_]vk.DynamicState{
            .viewport,
            .scissor
        };

        const dynamic_create_info = vk.PipelineDynamicStateCreateInfo{
            .dynamic_state_count = dynamic_states.len,
            .p_dynamic_states = &dynamic_states
        };

        const vertex_input_info = vk.PipelineVertexInputStateCreateInfo{
            .vertex_attribute_description_count = 0
        };

        const input_assembly = vk.PipelineInputAssemblyStateCreateInfo{
            .topology = .triangle_list,
            .primitive_restart_enable = vk.FALSE
        };

        const viewport_state = vk.PipelineViewportStateCreateInfo{
            .viewport_count = 1,
            .scissor_count = 1,
        };

        const rasterizer = vk.PipelineRasterizationStateCreateInfo{
            .depth_bias_enable = vk.FALSE,
            .depth_bias_constant_factor = 0.0,
            .depth_bias_clamp = 0.0,
            .depth_bias_slope_factor = 0.0,
            .rasterizer_discard_enable = vk.FALSE,
            .polygon_mode = .fill,
            .line_width = 1.0,
            .cull_mode = .{ .back_bit = true },
            .front_face = .clockwise,
            .depth_clamp_enable = vk.FALSE
        };

        const multisampling = vk.PipelineMultisampleStateCreateInfo{
            .sample_shading_enable = vk.FALSE,
            .rasterization_samples = .{ .@"1_bit" = true },
            .min_sample_shading = 1.0,
            .p_sample_mask = null,
            .alpha_to_one_enable = vk.FALSE,
            .alpha_to_coverage_enable = vk.FALSE
        };

        const color_blend_attachment = vk.PipelineColorBlendAttachmentState{
            .color_write_mask = .{
                .r_bit = true,
                .g_bit = true,
                .b_bit = true,
                .a_bit = true,
            },
            .blend_enable = vk.FALSE,
            .src_color_blend_factor = .one,
            .dst_color_blend_factor = .zero,
            .color_blend_op = .add,
            .src_alpha_blend_factor = .one,
            .dst_alpha_blend_factor = .zero,
            .alpha_blend_op = .add
        };

        const color_blending = vk.PipelineColorBlendStateCreateInfo{
            .logic_op_enable = vk.FALSE,
            .logic_op = .copy,
            .attachment_count = 1,
            .p_attachments = @ptrCast(&color_blend_attachment),
            .blend_constants = [_]f32{
                0.0, 0.0, 0.0, 0.0
            }
        };

        self.pipeline_layout = try self.device.createPipelineLayout(&.{}, null);

        const pipeline_info = vk.GraphicsPipelineCreateInfo{
            .stage_count = 2,
            .p_stages = &shader_stages,
            .p_vertex_input_state = &vertex_input_info,
            .p_input_assembly_state = &input_assembly,
            .p_viewport_state = &viewport_state,
            .p_rasterization_state = &rasterizer,
            .p_multisample_state = &multisampling,
            .p_depth_stencil_state = null,
            .p_color_blend_state = &color_blending,
            .p_dynamic_state = &dynamic_create_info,
            .layout = self.pipeline_layout,
            .render_pass = self.render_pass,
            .subpass = 0,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = -1
        };

        _ = try self.device.createGraphicsPipelines(
            .null_handle, 
            1, 
            @ptrCast(&pipeline_info),
            null, 
            @ptrCast(&self.graphics_pipeline)
        );
    }

    fn initFramebuffer(self: *GraphicalContext) !void {
        self.swapchain_framebuffers = std.ArrayList(vk.Framebuffer).init(self.allocator);

        for (self.images_view.items) |image_view| {
            const attachments = [_]vk.ImageView{ image_view };

            const framebuffer = try self.device.createFramebuffer(&.{
                .render_pass = self.render_pass,
                .attachment_count = 1,
                .p_attachments = &attachments,
                .width = self.swapchain_extent.width,
                .height = self.swapchain_extent.height,
                .layers = 1
            }, null);

            try self.swapchain_framebuffers.append(framebuffer);
        }
    }

    fn initCommandPool(self: *GraphicalContext) !void {
        const queue_families = try self.getQueueFamilies(self.physical_device);
        if (!queue_families.complete()) {
            return error.QueueNotComplete;
        }

        const pool_info = vk.CommandPoolCreateInfo{
            .flags = .{ .reset_command_buffer_bit = true },
            .queue_family_index = queue_families.graphics_family.?
        };
        
        self.command_pool = try self.device.createCommandPool(&pool_info, null);
    }

    fn initCommandBuffer(self: *GraphicalContext) !void {
        self.current_frame = 0;
        self.max_frames_in_flight = 3;
        const alloc_info = vk.CommandBufferAllocateInfo{
            .command_pool = self.command_pool,
            .level = .primary,
            .command_buffer_count = self.max_frames_in_flight,
        };

        var buffers: [*]vk.CommandBuffer = undefined;
        try self.device.allocateCommandBuffers(&alloc_info, @ptrCast(&buffers));
        print("hello\n", .{});
        self.frame_command_buffers = std.ArrayList(FrameCommandBuffer).init(self.allocator);
        for (buffers[0..self.max_frames_in_flight]) |cmd_buffer| {
            print("hello {?}\n", .{ cmd_buffer });

            const img, const render, const fence = try self.initSyncObjects();
            try self.frame_command_buffers.append(.{
                .command_buffer = cmd_buffer,
                .image_available_semaphore = img,
                .render_finished_semaphore = render,
                .in_flight_fence = fence
            });
        }

    }

    fn initSyncObjects(self: *GraphicalContext) !struct {vk.Semaphore, vk.Semaphore, vk.Fence} {
        const semaphore_info = vk.SemaphoreCreateInfo{};
        const fence_info = vk.FenceCreateInfo{
            .flags = .{ .signaled_bit = true }
        };

        return .{
            try self.device.createSemaphore(&semaphore_info, null),
            try self.device.createSemaphore(&semaphore_info, null),
            try self.device.createFence(&fence_info, null),
        };
    }

    fn recordCommandBuffer(self: *GraphicalContext, command_bufer: vk.CommandBuffer, image_index: u32) !void {
        try self.device.beginCommandBuffer(command_bufer, &.{});

        const render_pass_info = vk.RenderPassBeginInfo{
            .render_pass = self.render_pass,
            .framebuffer = self.swapchain_framebuffers.items[image_index],
            .render_area = .{
                .offset = .{ .x = 0.0, .y = 0.0 },
                .extent = self.swapchain_extent
            },
            .clear_value_count = 1,
            .p_clear_values = @ptrCast(&vk.ClearValue{
                .color = .{ .float_32 = [_]f32{0.0, 0.0, 0.0, 1.0} }
            }),
        };

        self.device.cmdBeginRenderPass(command_bufer, &render_pass_info, .@"inline");
        self.device.cmdBindPipeline(command_bufer, .graphics, self.graphics_pipeline);

        const viewport = vk.Viewport{
            .x = 0.0,
            .y = 0.0,
            .width = @floatFromInt(self.swapchain_extent.width),
            .height = @floatFromInt(self.swapchain_extent.height),
            .min_depth = 0.0,
            .max_depth = 1.0,
        };
        const scissor = vk.Rect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = self.swapchain_extent,
        };

        self.device.cmdSetViewport(command_bufer, 0, 1, @ptrCast(&viewport));
        self.device.cmdSetScissor(command_bufer, 0, 1, @ptrCast(&scissor));
        self.device.cmdDraw(command_bufer, 3, 1, 0, 0);

        self.device.cmdEndRenderPass(command_bufer);
        try self.device.endCommandBuffer(command_bufer);
    }

    pub fn drawFrame(self: *GraphicalContext) !void {
        const cur_frame_command_buffer = self.frame_command_buffers.items[self.current_frame];

        _ = try self.device.waitForFences(1, @ptrCast(&cur_frame_command_buffer.in_flight_fence), vk.TRUE, std.math.maxInt(u32));
        try self.device.resetFences(1, @ptrCast(&cur_frame_command_buffer.in_flight_fence));
            
        const acquire_res = try self.device.acquireNextImageKHR(
            self.swapchain, std.math.maxInt(u32), cur_frame_command_buffer.image_available_semaphore, .null_handle
        );

        try self.device.resetCommandBuffer(cur_frame_command_buffer.command_buffer, .{});
        try self.recordCommandBuffer(cur_frame_command_buffer.command_buffer, acquire_res.image_index);
        
        try self.device.queueSubmit(self.graphics_queue, 1, &[_]vk.SubmitInfo{.{
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&cur_frame_command_buffer.command_buffer),
            .wait_semaphore_count = 1,
            .p_wait_semaphores = @ptrCast(&cur_frame_command_buffer.image_available_semaphore),
            .signal_semaphore_count = 1,
            .p_signal_semaphores = @ptrCast(&cur_frame_command_buffer.render_finished_semaphore),
            .p_wait_dst_stage_mask = &[_]vk.PipelineStageFlags{.{.color_attachment_output_bit = true}},
        }}, cur_frame_command_buffer.in_flight_fence);

        _ = try self.device.queuePresentKHR(self.present_queue, &.{
            .wait_semaphore_count = 1,
            .p_wait_semaphores = @ptrCast(&cur_frame_command_buffer.render_finished_semaphore),
            .swapchain_count = 1,
            .p_swapchains = @ptrCast(&self.swapchain),
            .p_image_indices = @ptrCast(&acquire_res.image_index)
        });

        self.current_frame = (self.current_frame + 1) % self.max_frames_in_flight;
    }

    pub fn deviceWaitIdle(self: *GraphicalContext) !void {
        try self.device.deviceWaitIdle();
    }

    fn createShaderModule(self: *GraphicalContext, spirv_code: []u8) !vk.ShaderModule {
        return self.device.createShaderModule(&.{
            .code_size = spirv_code.len,
            .p_code = @alignCast(@ptrCast(spirv_code.ptr)),
        }, null);
    }

    fn getSuitableDevice(self: *GraphicalContext) !?vk.PhysicalDevice {
        const physical_devices = try self.instance.enumeratePhysicalDevicesAlloc(self.allocator);
        defer self.allocator.free(physical_devices);

        for (physical_devices) |device| {
            const prop = self.instance.getPhysicalDeviceProperties(device);
            const feats = self.instance.getPhysicalDeviceFeatures(device);

            const device_extensions = try self.instance.enumerateDeviceExtensionPropertiesAlloc(device, null, self.allocator);
            defer self.allocator.free(device_extensions);

            var extensions_count: u32 = 0;
            for (required_device_extensions) |required_extension| {
                for (device_extensions) |found_extension|  {
                    if (!std.mem.eql(u8, &makeLayerName(required_extension), &found_extension.extension_name)) continue;
                    extensions_count += 1;
                    break;
                }
            }

            const formats = try self.instance.getPhysicalDeviceSurfaceFormatsAllocKHR(device, self.surface, self.allocator);
            defer self.allocator.free(formats);

            const present_modes = try self.instance.getPhysicalDeviceSurfacePresentModesAllocKHR(device, self.surface, self.allocator);
            defer self.allocator.free(present_modes);

            const has_required_extensions = required_device_extensions.len == extensions_count;
            const is_discrete = prop.device_type == vk.PhysicalDeviceType.discrete_gpu;
            const has_geometry_shader = feats.geometry_shader == vk.TRUE;
            const is_swapchain_adequate = formats.len != 0 and present_modes.len != 0;

            try self.log(.INFO, .GENERAL, "Checking if device {s} is suitable:", .{prop.device_name});
            try self.log(.INFO, .GENERAL, "    Is Discrete: {any}", .{is_discrete});
            try self.log(.INFO, .GENERAL, "    Has Required Extensions: {any}", .{has_required_extensions});
            try self.log(.INFO, .GENERAL, "    Has Geometry Shader: {any}", .{has_geometry_shader});
            try self.log(.INFO, .GENERAL, "    Is SwapChain Adequate: {any}", .{is_swapchain_adequate});

            if (has_required_extensions and is_discrete and has_geometry_shader and is_swapchain_adequate) {
                try self.log(.INFO, .GENERAL, "Found suitable device: {s}", .{prop.device_name});
                return device;
            }
        }

        return null;
    }

    fn getQueueFamilies(self: *GraphicalContext, device: vk.PhysicalDevice) !QueueFamilies {
        const queue_props = try self.instance.getPhysicalDeviceQueueFamilyPropertiesAlloc(device, self.allocator);
        defer self.allocator.free(queue_props);

        var queue_families: QueueFamilies = .{};
        for (queue_props, 0..) |queue, index| {
            if (queue.queue_flags.graphics_bit) {
                queue_families.graphics_family = @intCast(index);
            }

            const presentation_support = try self.instance.getPhysicalDeviceSurfaceSupportKHR(device, @intCast(index), self.surface);
            if (presentation_support == vk.TRUE) {
                queue_families.presentation_family = @intCast(index);
            }

            if (queue_families.complete()) return queue_families;
        }

        return queue_families;
    }

    fn getRequiredExtensions(extensions: *std.ArrayList([*:0]const u8)) !void {
        var glfw_ext_count: u32 = 0;
        const glfw_ext = glfw.getRequiredInstanceExtension(&glfw_ext_count);

        try extensions.appendSlice(@ptrCast(glfw_ext[0..glfw_ext_count]));

        if (DEBUG) {
            try extensions.append(vk.extensions.ext_debug_utils.name);
        }
    }

    fn checkValidationLayerSupport(self: *GraphicalContext) !bool {
        var layer_count: u32 = undefined;
        _ = try self.vkb.enumerateInstanceLayerProperties(&layer_count, null);

        const available_layers = try self.allocator.alloc(vk.LayerProperties, layer_count);
        defer self.allocator.free(available_layers);

        _ = try self.vkb.enumerateInstanceLayerProperties(&layer_count, available_layers.ptr);

        for (validation_layers) |requested_layer| {
            var layer_found = false;

            for (available_layers) |layer| {
                layer_found = std.mem.eql(u8, &makeLayerName(requested_layer), &layer.layer_name);
                if (layer_found) {
                    return true;
                }
            }
        }

        return false;
    }

    fn setupDebugMessenger(self: *GraphicalContext) !void {
        if (!DEBUG) return;

        self.debug_messenger_ext = try self.instance.createDebugUtilsMessengerEXT(&initDebugCreateInfo(), null);
    }

    fn log(self: *GraphicalContext, severity: DebugSeverityFlag, log_type: DebugTypeFlag, comptime fmt: []const u8, args: anytype) !void {
        const message = try std.fmt.allocPrintZ(self.allocator, fmt, args);
        defer self.allocator.free(message);

        self.instance.submitDebugUtilsMessageEXT(
            vk.DebugUtilsMessageSeverityFlagsEXT.fromInt(@intFromEnum(severity)),
            vk.DebugUtilsMessageTypeFlagsEXT.fromInt(@intFromEnum(log_type)),
            &.{ .message_id_number = 0, .p_message = message }
        );
    }

    pub fn deinit(self: *GraphicalContext) void {
        if (self.debug_messenger_ext) |debug_ext| {
            self.instance.destroyDebugUtilsMessengerEXT(debug_ext, null);
        }

        for (self.frame_command_buffers.items) |f| {
            self.device.destroySemaphore(f.image_available_semaphore, null);
            self.device.destroySemaphore(f.render_finished_semaphore, null);
            self.device.destroyFence(f.in_flight_fence, null);
        }

        // std.debug.print("image view size: {d}\n", .{self.images_view.items.len});
        // const image_view_slice: []vk.ImageView = self.images_view.allocatedSlice();
        // std.debug.print("image view slice size: {d}\n", .{image_view_slice.len});
        self.device.destroyCommandPool(self.command_pool, null);
        self.frame_command_buffers.deinit();

        for (self.swapchain_framebuffers.items) |framebuffer| {
            self.device.destroyFramebuffer(framebuffer, null);
        }
        self.swapchain_framebuffers.deinit();

        for (self.images_view.items) |image_view| {
            self.device.destroyImageView(image_view, null);
        }

        self.device.destroyPipeline(self.graphics_pipeline, null);
        self.device.destroyPipelineLayout(self.pipeline_layout, null);
        self.device.destroyRenderPass(self.render_pass, null);

        self.images_view.deinit();
        self.allocator.free(self.images);

        self.device.destroySwapchainKHR(self.swapchain, null);
        self.instance.destroySurfaceKHR(self.surface, null);
        self.device.destroyDevice(null);
        self.instance.destroyInstance(null);
    }
};

