
class Solution:
    def fill(self, image, sr, sc, color, cur):
        R, C = len(image), len(image[0])
        directions = ((0, 1), (0, -1), (1, 0), (-1, 0))

        if sr < 0 or sr >= R or sc < 0 or sc >= C or cur != image[sr][sc]:
            return

        image[sr][sc] = color

        for d in directions:
            if sr < 0 or sr >= R or sc < 0 or sc >= C or cur != image[sr][sc]:
                self.fill(image, sr+d[0], sc+d[1], color, cur)

    def floodFill(self, image, sr, sc, color):
        if image[sr][sc] == color:
            return image

        self.fill(image, sr, sc, color, image[sr][sc])
        return image
