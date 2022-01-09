#include<iostream>
#include<string>
#include<vector>
#include<set>
using namespace std;
int Find(int x, vector<int> uf){
    if(uf[x]!=x){
        uf[x]=Find(uf[x],uf);
    }
    return uf[x];
}
bool Union(int x, int y, vector<int> &uf, vector<int>&rank){
    int px=Find(x,uf);
    int py=Find(y,uf);
    if(px==py) return false;
    else if(rank[px]<rank[py]) uf[px]=py;
    else if(rank[px]>rank[py]) uf[py]=px;
    else
        {
            uf[py] = px;
            ++rank[px];
        }
    return true;
}
int main(){
    int n,m;
    cin>>n>>m;
    vector<vector<int>> edges;
    for(int i=0;i<m;i++){
        int s,t,w;
        cin>>s>>t>>w;
        vector<int> temp={s,t,w,i};
        edges.push_back(temp);
    }
    vector<int>uf(n+1);
    vector<int>rank(n+1);
    for(int i=1;i<n+1;i++) {
        uf[i]=i;
        rank[i]=1;
    }
    sort(edges.begin(), edges.end(), [](const vector<int>& a, const vector<int>& b)
    {
        return a[2] < b[2];
    });
    int weights=0;
    vector<int> set;
    for(int i=0;i<m;i++){
        if(Union(edges[i][0],edges[i][1],uf,rank)) {
            weights+=edges[i][2];
            set.push_back(i);
        }
    }
    vector<int> res;
    for (int i = 0; i < m; ++i)
    {
        vector<int>uf1(n+1);
        vector<int>rank1(n+1);
        for(int i=1;i<n+1;i++) {
            uf1[i]=i;
            rank1[i]=1;
        }
        int w1=0;
        int n1=0;
        for (int j = 0; j < m; ++j)
        {
            if(i!=j && Union(edges[j][0],edges[j][1],uf1,rank1)) {
                w1+=edges[j][2];
                n1++;
            }
        }
        // 没有连通 或者 发现权重变大，那么就是关建边
        if (n1 != n-1 ||  (n1==n-1 && w1 > weights))
        {
            res.push_back(edges[i][3]);
        }
    }
    sort(res.begin(),res.end());
    for(int i=0;i<res.size();i++) cout<<res[i]<<endl;
}
