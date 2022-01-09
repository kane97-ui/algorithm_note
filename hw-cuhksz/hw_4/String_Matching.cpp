#include<iostream>
#include<string>
#include<vector>
using namespace std;
vector<int> compute_next(string P){
    int m=P.size();
    vector<int>next(m,-1);
    int k=-1;
    for(int q=1;q<m;q++){
        while(k>-1 and P[k+1]!=P[q]) k=next[k];
        if(P[k+1]==P[q]) k=k+1;
        next[q]=k;
    }
    return next;
}
int KMP_StringMatcher(string T,string P){
    int n=T.size();
    int m=P.size();
    vector<int> next=compute_next(P);
    int q=-1;
    for(int i=0;i<n;i++){
        while(q>-1 and P[q+1]!=T[i])q=next[q];
        if(P[q+1]==T[i]) q=q+1;
        if(q==m-1) return i-m+1;
    }
    return -1;
}
int main(){
    string T,P;
    cin>>T;
    cin>>P;
    int index=KMP_StringMatcher(T,P);
    cout<<index<<endl;
}