def bin_packing_bfd(items, bin_capacity):
    
    items.sort(reverse=True)

   
    bins = []
    bin_contents = []  

    for item in items:
        # Find the best bin for the item (min remaining capacity after placing it)
        min_space = float('inf')
        best_bin_index = None

        for i, remaining_capacity in enumerate(bins):
            if remaining_capacity >= item and (remaining_capacity - item) < min_space:
                min_space = remaining_capacity - item
                best_bin_index = i

        
        if best_bin_index is not None:
            bins[best_bin_index] -= item
            bin_contents[best_bin_index].append(item)
        else:
            
            bins.append(bin_capacity - item)
            bin_contents.append([item])

    return len(bins), bin_contents


# Example data
items = [30, 10, 20, 40, 35, 15, 25, 10, 20, 45, 5, 25, 15, 10, 30, 20, 40, 10]
bin_capacity = 50

# Solve the problem
num_bins, bin_contents = bin_packing_bfd(items, bin_capacity)

print(f"Minimum number of bins required: {num_bins}")
print("Bin contents:")
for i, contents in enumerate(bin_contents, 1):
    print(f"  Bin {i}: {contents}")
