const std = @import("std");
const glfw = @import("glfw.zig");
const vk = @import("vulkan");
const builtin = @import("builtin");

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

    std.debug.print(
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
    device: vk.DeviceProxy,
    surface: vk.SurfaceKHR,

    debug_messenger_ext: ?vk.DebugUtilsMessengerEXT,

    pub fn init(allocator: Allocator, window: *glfw.Window) !GraphicalContext {
        var self: GraphicalContext = undefined;

        self.allocator = allocator;
        self.vkb = vk.BaseWrapper.load(glfw.getInstanceProcAddress);

        try self.initInstance();
        try self.setupDebugMessenger();
        try self.initSurface(window);
        try self.initDevice();

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

            const has_required_extensions = required_device_extensions.len == extensions_count;
            const is_discrete = prop.device_type == vk.PhysicalDeviceType.discrete_gpu;
            const has_geometry_shader = feats.geometry_shader == vk.TRUE;

            try self.log(.INFO, .GENERAL, "Checking if device {s} is suitable:", .{prop.device_name});
            try self.log(.INFO, .GENERAL, "    Is Discrete: {any}", .{is_discrete});
            try self.log(.INFO, .GENERAL, "    Has Required Extensions: {any}", .{has_required_extensions});
            try self.log(.INFO, .GENERAL, "    Has Geometry Shader: {any}", .{has_geometry_shader});

            if (has_required_extensions and is_discrete and has_geometry_shader) {
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

        self.device.destroyDevice(null);
        self.instance.destroyInstance(null);
    }
};

