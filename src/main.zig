//! By convention, main.zig is where your main function lives in the case that
//! you are building an executable. If you are making a library, the convention
//! is to delete this file and start with root.zig instead.
const std = @import("std");
const glfw = @import("glfw.zig");
const vk = @import("vulkan");
const builtin = @import("builtin");

const Allocator = std.mem.Allocator;

pub const DEBUG = (builtin.mode == .Debug);

const validation_layers: [1][]const u8 = .{
    "VK_LAYER_KHRONOS_validation"
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

const GraphicalContext = struct {
    allocator: Allocator,

    vkb: vk.BaseWrapper,
    instance: vk.InstanceProxy,
    device: vk.DeviceProxy,

    debug_messenger_ext: ?vk.DebugUtilsMessengerEXT,

    pub fn init(allocator: Allocator) !GraphicalContext {
        var self: GraphicalContext = undefined;

        self.allocator = allocator;
        self.vkb = vk.BaseWrapper.load(glfw.getInstanceProcAddress);

        // ---------------------------------
        //  Instance Initialization
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

        const vki = try allocator.create(vk.InstanceWrapper);
        errdefer allocator.destroy(vki);

        vki.* = vk.InstanceWrapper.load(instance, self.vkb.dispatch.vkGetInstanceProcAddr.?);
        self.instance = vk.InstanceProxy.init(instance, vki);
        errdefer self.instance.destroyInstance(null);

        try self.setupDebugMessenger();

        // ---------------------------------
        //  Device Initialization
        const physical_devices = try self.instance.enumeratePhysicalDevicesAlloc(self.allocator);
        defer self.allocator.free(physical_devices);

        var choosen_device: ?vk.PhysicalDevice = null;
        var choosen_queue: ?usize = null;

        for (physical_devices) |device| {
            const prop = self.instance.getPhysicalDeviceProperties(device);
            const feats = self.instance.getPhysicalDeviceFeatures(device);
            
            if (prop.device_type != vk.PhysicalDeviceType.discrete_gpu or feats.geometry_shader == vk.FALSE) {
                continue;
            }

            const queue_families = try self.instance.getPhysicalDeviceQueueFamilyPropertiesAlloc(device, self.allocator);
            defer self.allocator.free(queue_families);

            for (queue_families, 0..) |queue, index| {
                if (!queue.queue_flags.graphics_bit) continue;
                choosen_queue = index;
                break;
            }

            try self.log(DebugSeverityFlag.INFO, DebugTypeFlag.GENERAL, "Using device: {s}\n", .{prop.device_name});
            choosen_device = device;
        }

        if (choosen_device == null) {
            return error.NoSuitableDeviceFound;
        }

        if (choosen_queue == null) {
            return error.NoSuitableQueueFound;
        }

        const device = self.instance.createDevice(choosen_device.?, &.{
            .p_queue_create_infos = [_]vk.DeviceQueueCreateInfo{
                .{
                    .queue_family_index = @intCast(choosen_queue.?),
                    .queue_count = 1,
                    .p_queue_priorities = [_]f32{ 1 }
                }
            },
        }, null);

        const vkd = try allocator.create(vk.DeviceWrapper);
        errdefer allocator.destroy(vkd);

        vkd.* = vk.DeviceWrapper.load(device, self.instance.dispatch.vkGetDeviceProcAddr.?);
        self.device = vk.DeviceProxy.init(device, vkd);
        errdefer self.device.destroyDevice(null);

        return self;
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

        self.instance.destroyInstance(null);
    }
};

fn getGlfwVersion() void {
    var major: i32 = 0;
    var minor: i32 = 0;
    var rev: i32 = 0;

    glfw.getVersion(&major, &minor, &rev);
    std.debug.print("GLFW {}.{}.{}\n", .{ major, minor, rev });
}


pub fn main() !void {
    getGlfwVersion();

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const allocator = arena.allocator();

    try glfw.init();
    defer glfw.terminate();

    var gl_ctx = try GraphicalContext.init(allocator);
    defer gl_ctx.deinit();

    // glfw.windowHint(glfw.ClientApi, glfw.NoApi);
    // glfw.windowHint(glfw.Resizable, 0);
    // glfw.initHint(glfw.Resizable, glfw.True);
    //
    // const window: *glfw.Window = try glfw.createWindow(800, 640, "Hello", null, null);
    // defer glfw.destroyWindow(window);
    //
    // while (!glfw.windowShouldClose(window)) {
    //     if (glfw.getKey(window, glfw.KeyEscape) == glfw.Press) {
    //         glfw.setWindowShouldClose(window, 1);
    //     }
    //
    //     glfw.pollEvents();
    // }
}

