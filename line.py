class Line:
    text = ""

    def __init__(self, rect):
        self.rect = rect

    def getY(self):
        return self.rect[1]

    def getX(self):
        return self.rect[0]

    def getHeight(self):
        return self.rect[3]

    def getWidth(self):
        return self.rect[2]
