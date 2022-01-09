#include<iostream>
#include<string>
#include<vector>
#include<set>
#include<queue>
using namespace std;
void dfs(int node, vector<vector<int>> &M,vector<int> &V, int &ft, priority_queue<pair<int,int>> &finish_time){
    V[node]=1;
    ft++;
    for(int i=0;i<M[node].size();i++){
        if(V[M[node][i]]) continue;
        dfs(M[node][i],M,V,ft,finish_time);
    }
    ft++;
    finish_time.push({ft,node});
}
void dfs_2(int node,vector<vector<int>> &M1,vector<int> &V1){
    V1[node]=1;
    for(int i=0;i<M1[node].size();i++){
        if(V1[M1[node][i]]) continue;
        dfs_2(M1[node][i],M1,V1);
    }
}
int main(){
    int n,m;
    cin>>n>>m;
    vector<vector<int>> M(n);
    vector<vector<int>> M1(n);
    for(int i=0;i<m;i++){
        int s,e;
        cin>>s>>e;
        M[s].push_back(e);
        M1[e].push_back(s);

    }
    vector<int>V(n,0);
    int ft=0;
    priority_queue<pair<int,int>> finish_time;
    for(int i=0;i<n;i++){
        if(V[i]) continue;
        dfs(i,M,V,ft,finish_time);
    }
    vector<int>V1(n,0);
    int num=0;
    while(!(finish_time.empty())){
        int node=finish_time.top().second;
        // cout<<finish_time.top().second<<" "<<finish_time.top().first<<endl;
        finish_time.pop();
        if(V1[node]) continue;
        num++;
        dfs_2(node,M1,V1);
    }
    cout<<num<<endl;

}