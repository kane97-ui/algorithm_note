#include<iostream>
#include<vector>
using namespace std;
int main(){
    int n,w;
    cin>>n>>w;
    vector<int> weight,value;
    int w_i,v_i;
    for(int i=0;i<n;i++){
        cin>>w_i>>v_i;
        weight.push_back(w_i);
        value.push_back(v_i);
    }
    vector<int> dp(w+1,0);
    vector<int> pre(w+1,0);
    for(int i=1;i<n+1;i++){
        if(i==n){
            if(weight[i-1]<=w) dp[w]=max(dp[w],pre[w-weight[i-1]]+value[i-1]);
            break;
        }
        for(int j=1;j<w+1;j++){
            if(weight[i-1]<=j) dp[j]=max(dp[j],pre[j-weight[i-1]]+value[i-1]);
        }
        pre=dp;
    }
    cout<< dp[w] <<endl;
}