import math


class FaceTracker:
    def __init__(self, alpha=0.35, max_missing=10):
        self.alpha = alpha
        self.max_missing = max_missing
        self.center = None
        self.box = None
        self.missing = 0
        self.active = False

    @staticmethod
    def get_box(face):
        b = face.bounding_box
        return (b.origin_x, b.origin_y, b.width, b.height)

    @staticmethod
    def get_center(box):
        x, y, w, h = box
        return (x + w / 2, y + h / 2)

    @staticmethod
    def iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - intersection

        return intersection / union if union > 0 else 0

    def update(self, detections):
        # 首次检测：选择面积最大的人脸
        if not self.active:
            if not detections:
                return None

            chosen = max(
                detections, key=lambda f: (f.bounding_box.width * f.bounding_box.height)
            )

            self.box = self.get_box(chosen)
            self.center = self.get_center(self.box)
            self.active = True
            self.missing = 0

            return self.center

        best_box = None
        best_score = -1

        # 寻找与上一帧目标最匹配的人脸
        for face in detections:
            box = self.get_box(face)
            center = self.get_center(box)

            overlap = self.iou(self.box, box)

            distance = math.dist(center, self.center)

            # 根据原目标大小确定匹配距离
            _, _, w, h = self.box
            max_distance = max(40, math.hypot(w, h) * 0.8)

            if overlap <= 0 and distance > max_distance:
                continue

            score = overlap + 0.3 * (1 - min(distance / max_distance, 1))

            if score > best_score:
                best_score = score
                best_box = box

        if best_box is None:
            self.missing += 1

            if self.missing > self.max_missing:
                self.active = False
                self.center = None
                self.box = None

            # 丢失时不输出新控制坐标
            return None

        self.missing = 0

        new_center = self.get_center(best_box)

        # 指数移动平均平滑
        self.center = (
            self.alpha * new_center[0] + (1 - self.alpha) * self.center[0],
            self.alpha * new_center[1] + (1 - self.alpha) * self.center[1],
        )

        self.box = best_box

        return self.center
