

# y = wx + b
def compute_error_for_line_give_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # computer mean-squared-totalError
        totalError += (y -(w * x + b)) ** 2
    # average loss for each point
    return totalError / float(len(points))
