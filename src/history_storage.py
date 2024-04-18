from datetime import datetime

class Message:
    def __init__(self, role, content, link):
        self.role = role
        self.content = content
        self.link = link
        self.timestamp = datetime.now()

class History:
    def __init__(self):
        self.messages = []

    def add_message(self, role, content, link=None):
        message = Message(role, content, link)
        self.messages.append(message)

    def get_history(self):
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def get_history_full(self):
        return [{"role": msg.role, "content": msg.content, "link": msg.link} for msg in self.messages]