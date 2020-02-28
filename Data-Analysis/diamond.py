class Diamond:
    def __init__(self, _id: int = 0, carat: float = 0, cut: str = "", color: str = "",
                 clarity: str = "", length: float = 0, width: float = 0, depth: float = 0,
                 table: float = 0, price: int = 0, depth_pct: float = 0):
        self.id: int = _id
        self.carat: float = carat
        self.cut: str = cut
        self.color: str = color
        self.clarity: str = clarity
        self.depth_pct: float = depth_pct
        self.table: float = table
        self.price: int = price
        self.length: float = length
        self.width: float = width
        self.depth: float = depth

    def parse(self, data: str):
        filtered = data.replace('"', '')
        columns = filtered.split(',')

        try:
            self.id = int(columns[0])
        except ValueError:
            self.id = None

        try:
            self.carat = float(columns[1])
        except ValueError:
            self.carat = None

        try:
            self.cut = columns[2]
        except ValueError:
            self.cut = None

        try:
            self.color = columns[3]
        except ValueError:
            self.color = None

        try:
            self.clarity = columns[4]
        except ValueError:
            self.clarity = None

        try:
            self.depth_pct = float(columns[5])
        except ValueError:
            self.depth_pct = None

        try:
            self.table = float(columns[6])
        except ValueError:
            self.table = None

        try:
            self.price = int(columns[7])
        except ValueError:
            self.price = None

        try:
            self.length = float(columns[8])
        except ValueError:
            self.length = None

        try:
            self.width = float(columns[9])
        except ValueError:
            self.width = None

        try:
            self.depth = float(columns[10])
        except ValueError:
            self.depth = None

        return self
