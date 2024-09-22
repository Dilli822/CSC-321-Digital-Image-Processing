def shape_number(chain_code):
    n = len(chain_code)
    rotations = [chain_code[i:] + chain_code[:i] for i in range(n)]
    return min(rotations)

# Compute the shape number from chain codes
shape_num = shape_number(chain_code)

# Visualize the shape number
print("Shape Number:", shape_num)
plt.plot(shape_num)
plt.title("Shape Number")
plt.show()
