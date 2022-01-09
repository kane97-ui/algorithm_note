#include<iostream>
#include<string>
#include<vector>
using namespace std;
int minDistance(string word1, string word2) {
    int n=word2.size();
    int m=word1.size();
    vector<int> dp(n+1,0);
    // for(int i=0;i<word1.size()+1;i++) dp[i][0]=i;
    for(int j=0;j<n+1;j++) dp[j]=j;
    int pre=0;
    for(int i=1;i<=m;i++){
        for(int j=0;j<=n;j++){
            int temp=dp[j];
            if(j==0){
                dp[j]=i;
            }
            else{
                int a= dp[j]+1;
                int b= dp[j-1]+1;
                int c=pre;
                if(word1[i-1]!=word2[j-1]) c++;
                dp[j]=min(a,min(b,c)); 
            } 
            pre=temp;
        }
    }
    return dp[n];
}
int main(){
    string A,B;
    cin>>A>>B;
    int distance=minDistance(A,B);
    cout<<distance<<endl;

}
