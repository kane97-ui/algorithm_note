#include <iostream>
#include <vector>
using namespace std;

#define MAXN 505

int link[MAXN], visx[MAXN], visy[MAXN],lx[MAXN],ly[MAXN];
int w[MAXN][MAXN];
int cost[MAXN][MAXN];
int n, m;
int MAX = 0xffffff;
int can(int t){
    visx[t] = 1;
    for(int i = 1; i <= m; i++){
        //这里“lx[t]+ly[i]==w[t][i]”决定了这是在相等子图中找增广路的前提，非常重要
        if(!visy[i] && lx[t] + ly[i] == w[t][i]){
            visy[i] = 1;
            if(link[i] == -1 || can(link[i])){
                link[i] = t;
                return 1;
            }
        }
    }
    return 0;
}

int km(){
    int sum = 0;
    for(int i=0; i<=n; i++)
        ly[i] = 0;
    for(int i = 1; i <= n; i++){//把各个lx的值都设为当前w[i][j]的最大值
        lx[i] = -MAX;
        for(int j = 1; j <= n; j++){
            if(lx[i] < w[i][j])
                lx[i] = w[i][j];
        }
    }
    for(int i=1; i<=n; i++)
        link[i] = -1;
    for(int i = 1; i <= n; i++){
        while(1){
            for(int k=0; k<=n; k++){
                visx[k] = 0;
                visy[k] = 0;
            }
            if(can(i))//如果它能够形成一条增广路径，那么就break
                break;
            int d = MAX;//否则，后面应该加入新的边,这里应该先计算d值
            //对于搜索过的路径上的XY点，设该路径上的X顶点集为S，Y顶点集为T，对所有在S中的点xi及不在T中的点yj
            for(int j = 1; j <= n; j++)
                if(visx[j]){
                    for(int k = 1; k <= m; k++)
                        if(!visy[k])
                            d = min(d, lx[j] + ly[k] - w[j][k]);
                }
            if(d == MAX)
                return -1;//找不到可以加入的边，返回失败（即找不到完美匹配）
            for (int j = 1; j <= n; j++){
                if (visx[j])
                    lx[j] -= d;
            }
            for(int j = 1; j <= m; j++){
                if(visy[j])
                    ly[j] += d;
            }
        }
    }
    for(int i = 1; i <= m; i++)
        if(link[i] > -1)
            sum += w[link[i]][i];
    return sum;
}

int connectTwoGroups() {
    int tn = n, tm = m;
    n = max(n, m);
    m = max(m, n);  //转换成方阵才能过   
    vector<int> lmin(tn + 1, MAX), rmin(tm + 1, MAX);
    for (int i = 1; i <= tn; ++i) {
        for (int j = 1; j <= tm; ++j) {
            lmin[i] = min(lmin[i], cost[i - 1][j - 1]);
            rmin[j] = min(rmin[j], cost[i - 1][j - 1]);
        }
    }
    int ans = 0;
    for(int i=1; i<=tn; i++)
        ans += lmin[i];
    for(int i=1; i<=tm; i++)
        ans += rmin[i];
    for (int i = 1; i <= tn; ++i) {
        for (int j = 1; j <= tm; ++j) {
            w[i][j] = max(0 , lmin[i] + rmin[j] - cost[i - 1][j - 1]);
        }
    }
    return ans - km();
}
int main()
{
    cin >> n >> m;
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            cin >> cost[i][j];
        }
    }
    cout << connectTwoGroups() << endl;;
    return 0;
}