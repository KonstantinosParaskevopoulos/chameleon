#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>


int ips_calc(int start, int end, int win_size, int step);
void max_min_calc(int ips, int start, int end, int win_size, int step);

int main(int argc, char *argv[]) {
    // Check number of arguments
    if (argc != 5) {
        return 1; // error
    }

    // Parse arguments (example: integers)
    int a = atoi(argv[1]);  // start
    int b = atoi(argv[2]);  // end
    int c = atoi(argv[3]);  // win_size
    int d = atoi(argv[4]);  // step

    // Do something (example computation)
    int max_ips = ips_calc(a, b, c, d);

    
    // Output ONLY the result
    printf("%d\n", max_ips);
    max_min_calc(max_ips, a,b,c,d); //Used Only for debug
    return 0;
}

void max_min_calc(int ips, int start, int end, int win_size, int step){
    int target;
    target = (start+end)/2;
    int min_target, max_target;
    int min_start, max_end;

    min_target = target - ((step * ips)/2);
    max_target = target + ((step * ips)/2);
    min_start = min_target - (win_size/2);
    max_end = max_target + (win_size/2);
    assert(min_start >= start && max_end <=end);
    printf("Real Start = %d\t|\tCalculated Start = %d\n", start, min_start);
    printf("Real End   = %d\t|\tCalculated End = %d\n", end, max_end);

}

int ips_calc (int start, int end, int win_size, int step){

    int max_ips;

    max_ips = (end-start-win_size)/step;
    max_ips = floor(max_ips);

    return max_ips;
}