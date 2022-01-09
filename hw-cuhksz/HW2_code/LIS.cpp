#include<iostream>
#include<vector>
using namespace std;
int main(){
    int n;
    cin>>n;
    vector<int> nums;
    int num;
    for(int i=0;i<n;i++) {
        cin>>num;
        nums.push_back(num);
    }
    if (n <= 1) return n;
    vector<int> dp;
    dp.push_back(nums[0]);
    for (int i = 1; i < n; ++i) {
        if (dp.back() < nums[i]) {
            dp.push_back(nums[i]);
        } else {
            /* 找到第一个比nums[i]大的元素 */
            auto itr = lower_bound(dp.begin(), dp.end(), nums[i]);
            *itr = nums[i];
        }
    }
    cout<<(int)dp.size()<<endl;
}
