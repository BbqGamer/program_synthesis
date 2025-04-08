#include <stdio.h>

int min(int a, int b);

int main() {
    int x, y, m;
    scanf("%d", &x);
    scanf("%d", &y);
    m = min(x, y);
    
    printf("Minimum of %d and %d is %d\n", x, y, m);
    
    return 1;
}

