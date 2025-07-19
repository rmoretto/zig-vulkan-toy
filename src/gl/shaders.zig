const std = @import("std");
const mem = std.mem;
const fs = std.fs;
const Allocator = mem.Allocator;

const SHADER_PATH = "assets/shaders";

const Stage = enum {
    vertex,
    fragment,
};

fn getShaderDir(allocator: Allocator) ![]u8 {
    var path_buffer: [fs.max_path_bytes]u8 = undefined;
    const exe_path = try fs.selfExeDirPath(&path_buffer);
    const exe_dir = fs.path.dirname(exe_path) orelse ".";

    return try fs.path.join(allocator, &[_][]const u8{exe_dir, SHADER_PATH});
}

pub fn compileShader(allocator: Allocator, shader_name: []const u8, stage: Stage) ![]u8 {
    const shader_dir = try getShaderDir(allocator);
    defer allocator.free(shader_dir);

    const shader_path = try fs.path.join(allocator, &[_][]const u8{shader_dir, shader_name});
    defer allocator.free(shader_path);

    const spirv_name = try std.fmt.allocPrint(allocator, "{s}_.spv", .{shader_name});
    defer allocator.free(spirv_name);

    const spirv_path = try fs.path.join(allocator, &[_][]const u8{shader_dir, spirv_name});
    defer allocator.free(spirv_path);

    const stage_name = switch (stage) {
        .vertex => "vert",
        .fragment => "fragment",
    };

    const shader_stage = try std.fmt.allocPrint(allocator, "-fshader-stage={s}", .{ stage_name });
    defer allocator.free(shader_stage);

    const glslc = try std.process.Child.run(.{
        .argv = &[_][]const u8{
            "glslc", 
            "--target-env=vulkan1.3", 
            shader_stage, 
            "-o", 
            spirv_path, 
            shader_path
        },
        .allocator = allocator
    });

    defer allocator.free(glslc.stdout);
    defer allocator.free(glslc.stderr);

    if (glslc.term != .Exited or glslc.term.Exited != 0) {
        std.debug.print("Error compiling shader {s}:\n", .{shader_name});
        std.debug.print("{s}:\n", .{glslc.stderr});
        return error.ShaderCompilerError;
    }

    const file = try fs.openFileAbsolute(spirv_path, .{});
    defer file.close();

    return try file.readToEndAlloc(allocator, 1024 * 1024);
}
