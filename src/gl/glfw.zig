const glfw_c = @cImport({
    @cDefine("GLFW_INCLUDE_NONE", {});
    @cInclude("GLFW/glfw3.h");
});

const vk = @import("vulkan");

pub const ClientApi = glfw_c.GLFW_CLIENT_API;
pub const NoApi = glfw_c.GLFW_NO_API;
pub const True = glfw_c.GLFW_TRUE;
pub const False = glfw_c.GLFW_FALSE;
pub const Resizable = glfw_c.GLFW_RESIZABLE;

pub const Window = glfw_c.GLFWwindow;
pub const Monitor = glfw_c.GLFWmonitor;

pub const destroyWindow = glfw_c.glfwDestroyWindow;
pub const getKey = glfw_c.glfwGetKey;
pub const getVersion = glfw_c.glfwGetVersion;
pub const initHint = glfw_c.glfwInitHint;
pub const pollEvents = glfw_c.glfwPollEvents;
pub const setWindowShouldClose = glfw_c.glfwSetWindowShouldClose;
pub const terminate = glfw_c.glfwTerminate;
pub const windowHint = glfw_c.glfwWindowHint;
pub const getRequiredInstanceExtension = glfw_c.glfwGetRequiredInstanceExtensions;

pub const getWindowUserPointer = glfw_c.glfwGetWindowUserPointer;
pub const setWindowUserPointer = glfw_c.glfwSetWindowUserPointer;
pub const setFramebufferSizeCallback = glfw_c.glfwSetWindowSizeCallback;

extern fn glfwGetInstanceProcAddress(instance: vk.Instance, procname: [*:0]const u8) vk.PfnVoidFunction;
extern fn glfwGetPhysicalDevicePresentationSupport(instance: vk.Instance, pdev: vk.PhysicalDevice, queuefamily: u32) c_int;
extern fn glfwCreateWindowSurface(instance: vk.Instance, window: *Window, allocation_callbacks: ?*const vk.AllocationCallbacks, surface: *vk.SurfaceKHR) vk.Result;

pub const getInstanceProcAddress = glfwGetInstanceProcAddress;
pub const getPhysicalDevicePresentationSupport = glfwGetPhysicalDevicePresentationSupport;
pub const createWindowSurface = glfwCreateWindowSurface;

pub const Key0 = glfw_c.GLFW_KEY_0;
pub const Key1 = glfw_c.GLFW_KEY_1;
pub const Key2 = glfw_c.GLFW_KEY_2;
pub const Key3 = glfw_c.GLFW_KEY_3;
pub const Key4 = glfw_c.GLFW_KEY_4;
pub const Key5 = glfw_c.GLFW_KEY_5;
pub const Key6 = glfw_c.GLFW_KEY_6;
pub const Key7 = glfw_c.GLFW_KEY_7;
pub const Key8 = glfw_c.GLFW_KEY_8;
pub const Key9 = glfw_c.GLFW_KEY_9;
pub const KeyA = glfw_c.GLFW_KEY_A;
pub const KeyApostrophe = glfw_c.GLFW_KEY_APOSTROPHE;
pub const KeyB = glfw_c.GLFW_KEY_B;
pub const KeyBackslash = glfw_c.GLFW_KEY_BACKSLASH;
pub const KeyBackspace = glfw_c.GLFW_KEY_BACKSPACE;
pub const KeyC = glfw_c.GLFW_KEY_C;
pub const KeyCaps_lock = glfw_c.GLFW_KEY_CAPS_LOCK;
pub const KeyComma = glfw_c.GLFW_KEY_COMMA;
pub const KeyD = glfw_c.GLFW_KEY_D;
pub const KeyDelete = glfw_c.GLFW_KEY_DELETE;
pub const KeyDown = glfw_c.GLFW_KEY_DOWN;
pub const KeyE = glfw_c.GLFW_KEY_E;
pub const KeyEnd = glfw_c.GLFW_KEY_END;
pub const KeyEnter = glfw_c.GLFW_KEY_ENTER;
pub const KeyEqual = glfw_c.GLFW_KEY_EQUAL;
pub const KeyEscape = glfw_c.GLFW_KEY_ESCAPE;
pub const KeyF = glfw_c.GLFW_KEY_F;
pub const KeyF1 = glfw_c.GLFW_KEY_F1;
pub const KeyF10 = glfw_c.GLFW_KEY_F10;
pub const KeyF11 = glfw_c.GLFW_KEY_F11;
pub const KeyF12 = glfw_c.GLFW_KEY_F12;
pub const KeyF13 = glfw_c.GLFW_KEY_F13;
pub const KeyF14 = glfw_c.GLFW_KEY_F14;
pub const KeyF15 = glfw_c.GLFW_KEY_F15;
pub const KeyF16 = glfw_c.GLFW_KEY_F16;
pub const KeyF17 = glfw_c.GLFW_KEY_F17;
pub const KeyF18 = glfw_c.GLFW_KEY_F18;
pub const KeyF19 = glfw_c.GLFW_KEY_F19;
pub const KeyF2 = glfw_c.GLFW_KEY_F2;
pub const KeyF20 = glfw_c.GLFW_KEY_F20;
pub const KeyF21 = glfw_c.GLFW_KEY_F21;
pub const KeyF22 = glfw_c.GLFW_KEY_F22;
pub const KeyF23 = glfw_c.GLFW_KEY_F23;
pub const KeyF24 = glfw_c.GLFW_KEY_F24;
pub const KeyF25 = glfw_c.GLFW_KEY_F25;
pub const KeyF3 = glfw_c.GLFW_KEY_F3;
pub const KeyF4 = glfw_c.GLFW_KEY_F4;
pub const KeyF5 = glfw_c.GLFW_KEY_F5;
pub const KeyF6 = glfw_c.GLFW_KEY_F6;
pub const KeyF7 = glfw_c.GLFW_KEY_F7;
pub const KeyF8 = glfw_c.GLFW_KEY_F8;
pub const KeyF9 = glfw_c.GLFW_KEY_F9;
pub const KeyG = glfw_c.GLFW_KEY_G;
pub const KeyGrave_accent = glfw_c.GLFW_KEY_GRAVE_ACCENT;
pub const KeyH = glfw_c.GLFW_KEY_H;
pub const KeyHome = glfw_c.GLFW_KEY_HOME;
pub const KeyI = glfw_c.GLFW_KEY_I;
pub const KeyInsert = glfw_c.GLFW_KEY_INSERT;
pub const KeyJ = glfw_c.GLFW_KEY_J;
pub const KeyK = glfw_c.GLFW_KEY_K;
pub const KeyKp_0 = glfw_c.GLFW_KEY_KP_0;
pub const KeyKp_1 = glfw_c.GLFW_KEY_KP_1;
pub const KeyKp_2 = glfw_c.GLFW_KEY_KP_2;
pub const KeyKp_3 = glfw_c.GLFW_KEY_KP_3;
pub const KeyKp_4 = glfw_c.GLFW_KEY_KP_4;
pub const KeyKp_5 = glfw_c.GLFW_KEY_KP_5;
pub const KeyKp_6 = glfw_c.GLFW_KEY_KP_6;
pub const KeyKp_7 = glfw_c.GLFW_KEY_KP_7;
pub const KeyKp_8 = glfw_c.GLFW_KEY_KP_8;
pub const KeyKp_9 = glfw_c.GLFW_KEY_KP_9;
pub const KeyKp_add = glfw_c.GLFW_KEY_KP_ADD;
pub const KeyKp_decimal = glfw_c.GLFW_KEY_KP_DECIMAL;
pub const KeyKp_divide = glfw_c.GLFW_KEY_KP_DIVIDE;
pub const KeyKp_enter = glfw_c.GLFW_KEY_KP_ENTER;
pub const KeyKp_equal = glfw_c.GLFW_KEY_KP_EQUAL;
pub const KeyKp_multiply = glfw_c.GLFW_KEY_KP_MULTIPLY;
pub const KeyKp_subtract = glfw_c.GLFW_KEY_KP_SUBTRACT;
pub const KeyL = glfw_c.GLFW_KEY_L;
pub const KeyLast = glfw_c.GLFW_KEY_LAST;
pub const KeyLeft = glfw_c.GLFW_KEY_LEFT;
pub const KeyLeft_alt = glfw_c.GLFW_KEY_LEFT_ALT;
pub const KeyLeft_bracket = glfw_c.GLFW_KEY_LEFT_BRACKET;
pub const KeyLeft_control = glfw_c.GLFW_KEY_LEFT_CONTROL;
pub const KeyLeft_shift = glfw_c.GLFW_KEY_LEFT_SHIFT;
pub const KeyLeft_super = glfw_c.GLFW_KEY_LEFT_SUPER;
pub const KeyM = glfw_c.GLFW_KEY_M;
pub const KeyMenu = glfw_c.GLFW_KEY_MENU;
pub const KeyMinus = glfw_c.GLFW_KEY_MINUS;
pub const KeyN = glfw_c.GLFW_KEY_N;
pub const KeyNum_lock = glfw_c.GLFW_KEY_NUM_LOCK;
pub const KeyO = glfw_c.GLFW_KEY_O;
pub const KeyP = glfw_c.GLFW_KEY_P;
pub const KeyPage_down = glfw_c.GLFW_KEY_PAGE_DOWN;
pub const KeyPage_up = glfw_c.GLFW_KEY_PAGE_UP;
pub const KeyPause = glfw_c.GLFW_KEY_PAUSE;
pub const KeyPeriod = glfw_c.GLFW_KEY_PERIOD;
pub const KeyPrint_screen = glfw_c.GLFW_KEY_PRINT_SCREEN;
pub const KeyQ = glfw_c.GLFW_KEY_Q;
pub const KeyR = glfw_c.GLFW_KEY_R;
pub const KeyRight = glfw_c.GLFW_KEY_RIGHT;
pub const KeyRight_alt = glfw_c.GLFW_KEY_RIGHT_ALT;
pub const KeyRight_bracket = glfw_c.GLFW_KEY_RIGHT_BRACKET;
pub const KeyRight_control = glfw_c.GLFW_KEY_RIGHT_CONTROL;
pub const KeyRight_shift = glfw_c.GLFW_KEY_RIGHT_SHIFT;
pub const KeyRight_super = glfw_c.GLFW_KEY_RIGHT_SUPER;
pub const KeyS = glfw_c.GLFW_KEY_S;
pub const KeyScroll_lock = glfw_c.GLFW_KEY_SCROLL_LOCK;
pub const KeySemicolon = glfw_c.GLFW_KEY_SEMICOLON;
pub const KeySlash = glfw_c.GLFW_KEY_SLASH;
pub const KeySpace = glfw_c.glfw_c.GLFW_KEY_SPACE;
pub const KeyT = glfw_c.GLFW_KEY_T;
pub const KeyTab = glfw_c.GLFW_KEY_TAB;
pub const KeyU = glfw_c.GLFW_KEY_U;
pub const KeyUp = glfw_c.GLFW_KEY_UP;
pub const KeyV = glfw_c.GLFW_KEY_V;
pub const KeyW = glfw_c.GLFW_KEY_W;
pub const KeyWorld_1 = glfw_c.GLFW_KEY_WORLD_1;
pub const KeyWorld_2 = glfw_c.GLFW_KEY_WORLD_2;
pub const KeyX = glfw_c.GLFW_KEY_X;
pub const KeyY = glfw_c.GLFW_KEY_Y;
pub const KeyZ = glfw_c.GLFW_KEY_Z;

pub const Press = glfw_c.GLFW_PRESS;
pub const Release = glfw_c.GLFW_RELEASE;

pub fn init() !void {
    if (glfw_c.glfwInit() == glfw_c.GLFW_FALSE) return error.GlfwInitError;
}

pub fn createWindow(width: c_int, height: c_int, title: [*:0]const u8, monitor: ?*Monitor, share: ?*Window) !*Window {
    const ret = glfw_c.glfwCreateWindow(width, height, title, monitor, share);
    if (ret == null) return error.GlfwInitWindowError;

    return ret.?;
}

pub fn windowShouldClose(window: *Window) bool {
    return glfw_c.glfwWindowShouldClose(window) == 1;
}


pub fn getFramebufferSize(window: *Window, width: *u32, height: *u32) void {
    var glfw_width: c_int = undefined;
    var glfw_height: c_int = undefined;
    glfw_c.glfwGetFramebufferSize(window, &glfw_width, &glfw_height);

    width.* = @intCast(glfw_width);
    height.* = @intCast(glfw_height);
}

