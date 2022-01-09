#include<iostream>
#include<vector>
#include<cmath>
#include<cstring>
using namespace std;
int main(){
    int n,m;
    cin>>n>>m;
    int dp[n];
    int nums[n];
    // vector<int> nums(n,0);
    // vector<int> dp(n,0);
    for(int i=0;i<n;i++){
        int num;
        cin>>num;
        nums[i]=num;
    }
    int res;
    int max_sum;
    // vector<int> mk(n,0);
    int mk[n];
    memset(dp,0,sizeof(int)*n);
    memset(mk,0,sizeof(int)*n);
    for(int i=0;i<m;i++){
        max_sum=INT_MIN;
        for(int j=i;j<n;j++){
            if(j==i) {
                if(j==0) dp[j]=nums[j];
                else dp[j]=mk[j-1]+nums[j];
            }
            else {
                dp[j]=max(dp[j-1]+nums[j],mk[j-1]+nums[j]);
                mk[j-1]=max_sum;
            }
            max_sum=max(max_sum,dp[j]);
        }
    }
    cout<<max_sum<<endl;

}