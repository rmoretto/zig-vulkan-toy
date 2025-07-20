const std = @import("std");
const glfw = @import("gl/glfw.zig");
const gl_ctx = @import("gl/ctx.zig");

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


    glfw.windowHint(glfw.ClientApi, glfw.NoApi);
    glfw.windowHint(glfw.Resizable, 0);
    glfw.initHint(glfw.Resizable, glfw.True);

    const window: *glfw.Window = try glfw.createWindow(800, 640, "Hello", null, null);
    defer glfw.destroyWindow(window);

    var ctx = try gl_ctx.GraphicalContext.init(allocator, window);
    defer ctx.deinit();

    while (!glfw.windowShouldClose(window)) {
        if (glfw.getKey(window, glfw.KeyEscape) == glfw.Press) {
            glfw.setWindowShouldClose(window, 1);
            continue;
        }

        try ctx.drawFrame();
        glfw.pollEvents();
    }

    try ctx.deviceWaitIdle();
}

