for idx, (product, sales) in enumerate(results['top_products']['Sales'].items(), 1):
    try:
        # Assuming `append_func` is defined somewhere
        append_func(product, sales)
    except Exception as e:
        print(f'Error processing product {product}: {e}')

# (Keep the other code from lines 1104 to 1122 unchanged)

for idx, (product, sales) in enumerate(results['slow_products']['Sales'].items(), 1):
    # Your logic here
