class BusinessException(Exception):
    def __init__(self, message, error_code):
        super().__init__(message)  # 调用基类的构造函数
        self.error_code = error_code  # 添加自定义属性

    def __str__(self):
        # 定义异常的字符串表示方式
        return f"{super().__str__()}"