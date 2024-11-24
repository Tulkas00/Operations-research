def bin_packing_ffd(items, bin_capacity):
   
    items.sort(reverse=True)
    bins = []
    bin_contents = []
    
    for item in items:
        placed = False
        
        for i in range(len(bins)):
            if bins[i] >= item:
                bins[i] -= item
                bin_contents[i].append(item)
                placed = True
                break
        if not placed:
            bins.append(bin_capacity - item)
            bin_contents.append([item])
    
    return len(bins), bin_contents


# Example data
items = [30, 10, 20, 40, 35, 15, 25, 10, 20, 45, 5, 25, 15, 10, 30, 20, 40, 10]
bin_capacity = 50

# Solve the problem
num_bins, bin_contents = bin_packing_ffd(items, bin_capacity)

print(f"Minimum number of bins required: {num_bins}")
print("Bin contents:")
for i, contents in enumerate(bin_contents, 1):
    print(f"  Bin {i}: {contents}")
