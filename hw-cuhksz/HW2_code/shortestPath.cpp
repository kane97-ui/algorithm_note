#include<iostream>
#include<vector>
#include<cmath>
#include<climits>
using namespace std;
int main(){
    int n,m,s,t;
    cin>>n>>m>>s>>t;
    vector<vector<int>> edges;
    for(int i=0;i<m;i++){
        int u,v,w;
        cin>>u>>v>>w;
        edges.push_back({u,v,w});
    }
    vector<int> dst(n+1,INT_MAX);
    dst[s]=0;
    for(int i=0;i<n-1;i++){
        int flag=0;
        for(const auto& e:edges){
            int u=e[0];
            int v=e[1];
            int w=e[2];
            if(dst[u]!=INT_MAX && dst[v]>dst[u]+w){
                dst[v]=dst[u]+w;
                flag=1;
            }
        }
        if(!flag) break;

    }
    cout<<dst[t]<<endl;
}