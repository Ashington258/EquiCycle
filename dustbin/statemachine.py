import time


class StateMachine:
    """状态机类，执行任务代码"""

    def __init__(self):
        self.state = "IDLE"  # 初始状态
        self.executed_tasks = {
            "zebra": False,  # 斑马线任务是否执行过
            "turn_sign": False,  # 转向标志任务是否执行过
        }
        self.cone_count = 0  # 记录锥桶的检测次数
        self.cone_task_executed = False  # 锥桶任务是否执行
        self.cone_start_time = None  # 锥桶置信度满足条件的开始时间
        self.cone_detection_active = False  # 当前是否检测到锥桶
        self.cone_confidence_met = False  # 锥桶置信度是否已达到条件

    def transition(self, detections, confidence_threshold=0.85):
        """
        根据检测结果进行状态转换并执行任务。
        detections: List[str] - 检测的类型列表（如 ["cone", "zebra"]）
        """
        current_time = time.time()  # 当前时间戳

        # 判断是否检测到锥桶
        if "cone" in detections:
            self.handle_cone_detection(current_time)
        else:
            # 如果没有检测到锥桶，重置相关状态
            self.reset_cone_detection()

        # 处理其他任务
        for detection in detections:
            if detection == "zebra" and not self.executed_tasks["zebra"]:
                # 如果检测到斑马线且任务未执行，则优先执行斑马线任务
                self.state = "DETECTED_ZEBRA"
                self.execute_task2()
            elif detection == "turn_sign" and not self.executed_tasks["turn_sign"]:
                # 转向任务仅在斑马线任务已执行的情况下才会执行
                if self.executed_tasks["zebra"]:  # 依赖于斑马线任务完成
                    self.state = "DETECTED_TURN_SIGN"
                    self.execute_task3()
            else:
                # 如果没有未执行的任务，回到空闲状态
                self.state = "IDLE"

    def handle_cone_detection(self, current_time):
        """处理锥桶检测逻辑"""
        if not self.cone_confidence_met:
            # 如果是首次达到置信度阈值，记录开始时间
            self.cone_start_time = current_time
            self.cone_confidence_met = True
            print("Cone detection started. Waiting for 3 seconds...")
        else:
            # 检查置信度是否持续超过3秒
            if (
                current_time - self.cone_start_time >= 3
                and not self.cone_detection_active
            ):
                # 确认锥桶检测成功，并标记为“活动检测状态”
                self.cone_detection_active = True
                self.cone_count += 1  # 每次有效检测到锥桶时计数加1

                if self.cone_count < 3:
                    # 第一次和第二次检测到锥桶时，仅显示锥桶序号
                    print(
                        f"Detected cone number {self.cone_count}. Waiting for the third cone..."
                    )
                else:
                    # 第三次检测到锥桶时，执行任务1
                    print("Detected third cone. Executing Task 1...")
                    self.execute_task1()
                    self.cone_task_executed = True  # 标记锥桶任务已执行

    def reset_cone_detection(self):
        """重置锥桶检测状态"""
        if self.cone_detection_active:
            print("Cone detection lost. Resetting detection state...")
        self.cone_confidence_met = False  # 重置置信度状态
        self.cone_detection_active = False  # 重置活动检测状态
        self.cone_start_time = None  # 重置计时器

    def execute_task1(self):
        """执行锥桶检测任务"""
        print("Task 1: Detected a cone. Executing Task 1...")

    def execute_task2(self):
        """执行斑马线检测任务"""
        print("Task 2: Detected a zebra crossing. Executing Task 2...")
        self.executed_tasks["zebra"] = True

    def execute_task3(self):
        """执行转向标志检测任务"""
        print("Task 3: Detected a turn sign. Executing Task 3...")
        self.executed_tasks["turn_sign"] = True

    def reset_tasks(self):
        """重置任务状态"""
        for task in self.executed_tasks:
            self.executed_tasks[task] = False
        self.cone_count = 0  # 重置锥桶计数
        self.cone_task_executed = False
        self.cone_start_time = None
        self.cone_detection_active = False
        self.cone_confidence_met = False

    def get_state(self):
        """返回当前状态"""
        return self.state
