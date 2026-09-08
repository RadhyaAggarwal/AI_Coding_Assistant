def print_star(n):
    for i in range(n):
        # Print leading spaces
        for j in range(i, n-1):
            print(' ', end='')
        # Print stars
        for k in range(2*i+1):
            print('*', end='')
        print()

if __name__ == '__main__':
    print_star(5)