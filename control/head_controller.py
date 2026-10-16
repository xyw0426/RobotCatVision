class HeadController:
    def __init__(
        self,
        yaw_limit=(-60, 60),
        pitch_limit=(-20, 30),
        gain=45.0,
        deadzone=0.06,
        max_speed=60.0,
    ):
        self.yaw = 0.0
        self.pitch = 0.0

        self.yaw_limit = yaw_limit
        self.pitch_limit = pitch_limit

        self.gain = gain
        self.deadzone = deadzone
        self.max_speed = max_speed

    @staticmethod
    def clamp(value, low, high):
        return max(low, min(high, value))

    def update(self, error_x, error_y, dt):
        # 限制时间步长，避免程序卡顿后突然转头
        dt = self.clamp(dt, 0.0, 0.1)

        # 设置死区：靠近画面中心时停止调整
        if abs(error_x) < self.deadzone:
            error_x = 0.0

        if abs(error_y) < self.deadzone:
            error_y = 0.0

        # 控制速度（度/秒）
        yaw_speed = self.clamp(self.gain * error_x, -self.max_speed, self.max_speed)

        pitch_speed = self.clamp(self.gain * error_y, -self.max_speed, self.max_speed)

        # 更新当前虚拟角度
        self.yaw = self.clamp(self.yaw + yaw_speed * dt, *self.yaw_limit)

        self.pitch = self.clamp(self.pitch + pitch_speed * dt, *self.pitch_limit)

        return self.yaw, self.pitch
