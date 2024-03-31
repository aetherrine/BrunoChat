from datetime import datetime

class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content
        self.timestamp = datetime.now()

class History:
    def __init__(self):
        self.messages = []

    def add_message(self, role, content):
        message = Message(role, content)
        self.messages.append(message)

    def get_history(self):
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]