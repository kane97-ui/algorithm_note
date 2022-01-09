#include<iostream>
#include<vector>
#include<cstdio>
using namespace std;
int main(){
    int num;
    vector<int> seq1,seq2;
    while(scanf("%d",&num)) {
        seq1.push_back(num);
        if(cin.get()=='\n') break;
    }
    while(scanf("%d",&num)){
        seq2.push_back(num);
        if(cin.get()=='\n') break;
    }
    int n=seq1.size();
    vector<int> dp(n+1,0);
    int pre;
    for(int i=1;i<n+1;i++){
        for(int j=1;j<n+1;j++){
            int temp=dp[j];
            if(i==1 && j==1) dp[j]=seq1[i-1]==seq2[j-1]?1:0;
            else if(i==1)  dp[j]=seq1[i-1]==seq2[j-1]?1:dp[j-1];
            else if(j==1)  dp[j]=seq1[i-1]==seq2[j-1]?1:dp[j];
            else{
                if(seq1[i-1]==seq2[j-1]) dp[j]=pre+1;
                else dp[j]=max(dp[j-1],dp[j]);
            }
            pre=temp;
        }
    }
    cout<<dp[n]<<endl;
}