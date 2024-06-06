x = [1, 2, 3]
y = [5, 1, 3]

#y = x
m1 = 1
b1 = 0

#y = 0.5x + 1
m2 = 0.5
b2 = 1

# Find the y-values that the line with weights m1 and b1 would predict for the x-values given. Store these in a list called y_predicted1.
# y = mx + b
y_predicted1 = [m1 * xi + b1 for xi in x]

# Find the y values that the line with weights m2 and b2 would predict for the x-values given. Store these in a list called y_predicted2.
y_predicted2 = [m2 * xi + b2 for xi in x]

print("Predicted y-values for line 1:", y_predicted1)
print("Predicted y-values for line 2:", y_predicted2)

#  find the sum of the squared distance between the actual y-values of the points and the y_predicted1 values
total_loss1 = 0
for yi, y_pred in zip(y, y_predicted1):
  difference = yi - y_pred
  squared_difference = difference ** 2
  total_loss1 += squared_difference

print("Total loss:", total_loss1)

# sum of the squared distance between the actual y-values of the points and the y_predicted2 values
total_loss2 = 0
for i in range(len(y)):
  total_loss2 += (y[i] - y_predicted2[i]) ** 2

print("Total loss 2:", total_loss2)

# Print out total_loss1 and total_loss2. Out of these two lines, which would you use to model the points
# The line that produces the lowest total loss will be the line of better fit.
if total_loss2 < total_loss1:
  better_fit = 2
else:
  better_fit = 1




# GRADIENT DESCENT
# we find the sum of y_value - (m*x_value + b) for all the y_values and x_values we have
# and then we multiply the sum by a factor of -2/N. N is the number of points we have.

def get_gradient_at_b(x, y, m, b):
  diff = 0
  # N is the number of points
  N = len(x)
  for i in range(0, len(x)):
    y_val = y[i]
    x_val = x[i]
    diff += (y_val - ((m * x_val) + b))
  # Define b_gradient
  b_gradient = -2/N * diff
  return b_gradient

# ALTERNATIVE
# def get_gradient_at_b(x, y, m, b):
#   diff = 0
#   N = len(x)
#   for x_val, y_val in zip(x, y):
#     diff += y_val - (m*x_val +b)

#   b_gradient = -2 / N * diff  

#   return b_gradient
