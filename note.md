# Array

## 一个数组中，三个数的和的问题或者四个数的和的问题
（1）首先将数组排序。可以通过先定第一个数，然后通过双指针，分别指向第一个数的后一个元素，和最后一个元素，这样问题就变成了双数之和的问题。通过判断后两个数的和的大小，移动指针：1.如果过小，left指针右移；2.如果过大，right指针左移。

```cpp
for(int i=0;i<nums.size()-2;i++){
          if (nums[i]>0) return res;
          if (i>0 && nums[i]==nums[i-1]) continue;
          int left=i+1;
          int right=nums.size()-1;
          while (right>left){
              if (nums[left]+nums[right]==-nums[i]) {
                  res.push_back(vector<int>{nums[i],nums[left],nums[right]});
                  while(left<right && nums[left+1]==nums[left]) left++;
                  while(left<right && nums[right-1]==nums[right]) right--;}
              if (nums[left]+nums[right]<-nums[i]) left+=1;
              else right-=1;
          }
```
  就办法是用三重循环，每个循环遍历一遍数组，这样的时间复杂度是**O($n^3$)**,但如果用上述方法，时间复杂度是**O($n^2$)**。例题：15，16.

## 一个数组中，两个数的运算的问题
和三个数的和的问题相识，也是通过双指针的方法。

## 组合问题
对于数组中的组合问题，可以使用递归的方法。例：**39**

```cpp
   //组合问题1:无重复数组，且每个元素可以重复使用
   void dfs(int target,vector<int>& candidates,vector<vector<int>>& combine, vector<int> &temp,int idx,int size ){
      if (target==0){
          combine.push_back(temp);
          return;
      }
      if (idx>=size || target<0) return;
      temp.push_back(candidates[idx]);
      // vector<int> temp1=temp;
      // vector<int> temp2=temp;
      dfs(target-candidates[idx],candidates,combine, temp,idx,size );
      temp.pop_back();
      // cout<<*temp.begin()<<*temp.end()<<endl;
      dfs(target,candidates,combine, temp,idx+1,size );
      return;
  //组合问题2:有重复的数组，并且每个元素不可以重复用。可以看作排列问题。这里的candidates是排好序的数组，在每一个阶段的回溯不取一样的值。
      void back_tracking(vector<int>& candidates,vector<vector<int>>& save,vector<int> temp,int target,int index){
      if (target==0 && find(save.begin(),save.end(),temp)==save.end()) {save.push_back(temp);return;}
      if(target<0) return;
      for(int i=index;i<candidates.size();i++){
              if (i>index && candidates[i]==candidates[i-1]) continue;
              temp.push_back(candidates[i]);
              back_tracking(candidates,save,temp,target-candidates[i],i+1);
              temp.pop_back();
      }
```

# Tree

## 树的前中后序遍历
### 用递归的方式：

```cpp
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        inorder(root,res);
        return res;
    }
    void inorder(TreeNode *root,vector<int>& res){
        if (root==nullptr){
                return;
        }
        //前序遍历
        res.push_back(root->val);
        inorder(root->left,res);
        inorder(root->right,res);
        /***中序遍历
        inorder(root->left,res);
        res.push_back(root->val);
        inorder(root->right,res);
        ***/
       /***后序遍历
        inorder(root->left,res);
        inorder(root->right,res);
        res.push_back(root->val);
       ***/
    }
};
```
### 用栈迭代方式：

```cpp
class Solution {
public:
    //中序遍历
    vector<int> inorderTraversal(TreeNode* root) {
        stack<TreeNode*> stk;
        vector<int> res;
        if (!root) return res;
        stk.push(root);
        while(!stk.empty()){
            TreeNode * node=stk.top();
            if (node->left) {stk.push(node->left);node->left=NULL;
            continue;}
            res.push_back(node->val);
            stk.pop();
            if (node->right) {stk.push(node->right);node->right=NULL;continue;}
        }
        return res;
    }
    /**后序遍历
        vector<int> postorderTraversal(TreeNode* root) {
        stack<TreeNode*> stk;
        vector<int> res;
        if (!root) return res;
        stk.push(root);
        while(!stk.empty()){
            TreeNode * node=stk.top();
            if (node->left) {stk.push(node->left);node->left=NULL;
            continue;}
            if (node->right) {stk.push(node->right);node->right=NULL;continue;}
            stk.pop();
            res.push_back(node->val);
        }
        return res;
     **/
    /**
        vector<int> preorderTraversal(TreeNode* root) {
     stack<TreeNode*> stk;
        vector<int> res;
        if (!root) return res;
        stk.push(root);
        res.push_back(root->val);
        while(!stk.empty()){
            TreeNode * node=stk.top();
            if (node->left) {stk.push(node->left);res.push_back(node->left->val);node->left=NULL;
            continue;}
            if (node->right) {stk.push(node->right);res.push_back(node->right->val);node->right=NULL;continue;}
            stk.pop();
        }
        return res;
        **/

 }
};
```

## 树的层级遍历和求树的深度

这种情况下，最好用队列:

```cpp
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> result;
        queue<TreeNode*> q;
        vector<int> temp;
        if (!root) return result;
        q.push(root);
        q.push(NULL);
        while(!q.empty()){
            TreeNode * t=q.front();
            q.pop();
            if (!t) {
                result.push_back(temp);
                temp.resize(0);
                if (q.size() > 0) {
                    q.push(NULL);
                }
            } 
            else{
            temp.push_back(t->val);
            if (t->left!=NULL) q.push(t->left);
            if (t->right!=NULL) q.push(t->right);
            }
        }
        return result;
    };
```

## 二叉搜索树

### 数组转换成二叉搜索树

二叉搜索数的中序遍历是升序序列，选择中间的数字作为二叉搜索树的根结点，可以使得树保持平衡。

```cpp
    TreeNode* help(vector<int>& nums,int left,int right){
      if (left>right) {return nullptr;}
       int mid=(left+right)/2;
       TreeNode * node=new TreeNode(nums[mid]);
       node->left=help(nums,left,mid-1);
       node->right=help(nums,mid+1,right);
       return node;
  }  
```

### 验证二叉搜索树
这里用dfs，但和一般的回溯不一样，例如平衡二叉树的检验，是从底回溯到顶进行检验。这里在每一次向后递归都会检验是否满足二叉搜索树，不满足直接返回false，否组继续往后递归至树底，最后回溯到树顶。

```cpp
class Solution {
  public:
  bool isValidBST(TreeNode* root) {
      return dfs(root,LONG_MIN,LONG_MAX);

  }
  bool dfs(TreeNode * root,long long lower, long long upper ){
      if (root == nullptr) {
          return true;
      }
      if (root -> val <= lower || root -> val >= upper) {
          return false;
      }
      return dfs(root -> left, lower, root -> val) && dfs(root -> right, root -> val, upper);
  } 
 };
```

### 平衡二叉树 
方法一：自顶向下：
**$height(p)={^{0}_{max(height(p,left),height(p.right))+1}}$**      **$^{p是空节点}_{p是非空节点}$**
此方法类似二叉树的前序遍历，首先计算左右子树的高度，如果不超过1.再分别递归地遍历左右子节点，并判断左子树和右子树是否平衡.

* 时间复杂度：$O(n^2)$,n是二叉树中的结点个数。
* 空间复杂度： $O(n)$，其中n是二叉树中的节点个数。空间复杂度主要取决于递归调用的层数，递归调用的层数不会超过n.
  ```cpp  
  class Solution {
  public:
      int height(TreeNode* root) {
          if (root == NULL) {
              return 0;
          } else {
              return max(height(root->left), height(root->right)) + 1;
          }
      }
  
      bool isBalanced(TreeNode* root) {
          if (root == NULL) {
              return true;
          } else {
              return abs(height(root->left) - height(root->right)) <= 1 && isBalanced(root->left) && isBalanced(root->right);
          }
      }
  };
  ```
   方法二：自底向上的递归：自底向上递归的做法类似于后序遍历，对于当前遍历到的节点，先递归地判断其左右子树是否平衡，再判断以当前节点为根的子树是否平衡。
   ```cpp
   class Solution {
    public:
        int height(TreeNode* root) {
            if (root == NULL) {
                return 0;
            }
            int leftHeight = height(root->left);
            int rightHeight = height(root->right);
            if (leftHeight == -1 || rightHeight == -1 || abs(leftHeight - rightHeight) > 1) {
                return -1;
            } else {
                return max(leftHeight, rightHeight) + 1;
            }
        }
  
        bool isBalanced(TreeNode* root) {
            return height(root) >= 0;
        }
    };
   ```
* 时间复杂度：$O(n)$,n是二叉树中的结点个数。使用自底向上的递归，每个节点的计算高度和判断是否平衡都只需要处理一次，最坏情况下需要遍历二叉树中的所有节点，
* 空间复杂度： $O(n)$，其中n是二叉树中的节点个数.空间复杂度主要取决于递归调用的层数，递归调用的层数不会超过 

# 动态规划

将一个问题拆成几个子问题，分别求解这些子问题，即可推断出大问题的解。
例：我们记“凑出n所需的最少钞票数量”为f(n).
　　f(n)的定义就已经蕴含了“最优”。利用w=14,10,4的最优解，我们即可算出w=15的最优解。大问题的最优解可以由小问题的最优解推出，这个性质叫做“最优子结构性质”。
**步骤：**

* 1.如果n=15，我们设计一个长度为15的数组，初始化它。  
* 2.每个数组保存一个n=i的状态，而n=i+1的状态是由n=0-i的状态决定的
* 3.n=15则是我们要求的最优解。

**例题：**

## 最大子序和

```cpp
int maxSubArray(vector<int>& nums) {
        int size=nums.size();
        vector<int> save;
        // save中第i个元素保存了，以第i个数字为结尾最大子序和。
        for (int i=0;i<size;i++){
            if (i==0) save.push_back(nums[i]);
            else{
                if (nums[i]<(save[i-1]+nums[i])) save.push_back(save[i-1]+nums[i]);
                else save.push_back(nums[i]);
            }
        }
        auto maxvalue=max_element(save.begin(),save.end());
        return *maxvalue;

    }
```
# BackTracking

## 排列问题，一般会在回归函数中加入循环.注意：当回溯完一遍，会pop出最近进来的元素。

```cpp
//例1:全排列问题
    void back_tracking(vector<int> & nums,vector<vector<int>> & save,vector<int> temp){
      if (temp.size()==nums.size()) {save.push_back(temp);return;}
      for (int i =0;i<nums.size();i++){
          if (find(temp.begin(),temp.end(),nums[i])==temp.end()) {temp.push_back(nums[i]);
          back_tracking(nums,save,temp);}
          else continue;
          temp.pop_back();

      }
  //例2:电话号码的字母组合
  void back_tracking(string digits,vector<string>& save,int size,int index,string combination,map<char,string>mp){
            if (index==size){
                save.push_back(combination);
                return;
            }
            char number=digits[index];
            string characters=mp[number];
            for(char &letter: characters){
                 combination=combination+letter;
                 back_tracking(digits,save,size,index+1,combination,mp);
                 combination.pop_back();
            }
  }
  //例3:N皇后问题
      class Solution {
  public:
      vector<vector<string>> solveNQueens(int n) {
          vector<vector<int>> state(n,vector<int>(n,0));
          vector<vector<string>> save;
          vector<string> temp;
          back_tracking(n,state,save,temp,0,0);
          return save;

      }
      void back_tracking(int n, vector<vector<int>> & state,vector<vector<string>> & save,vector<string> temp,int i,int j){
          for(int row=1;row<=i-1;row++){
              if(state[i-1-row][j]) return;
              if (j>=row and state[i-1-row][j-row]) return;
              if (j+row<n and state[i-1-row][j+row]) return;
          }
          if (temp.size()==n) {save.push_back(temp);return;}
          for(int col=0;col<n;col++){
              state[i][col]=1;
              string str(n,'.');
              str[col]='Q';
              temp.push_back(str);
              // cout<<str<<endl;
              back_tracking(n,state,save,temp,i+1,col);
              state[i][col]=0;
              temp.pop_back();
          }

      }
  };
```

## 树的深度优先问题，这种情况一般是用栈或者回溯法。深度优先一般是采用树的前序遍历。

```cpp 
//路径问题i
bool dfs(TreeNode * root,int targetSum){
      if(!root) return false;
      else targetSum-=root->val;
      if(!root->left and !root->right) {
          if(targetSum==0) return true;
          else return false;
      }
      return dfs(root->left,targetSum)||dfs(root->right,targetSum);
      
  }
  //路径问题ii
  void dfs(TreeNode * root, int targetSum,vector<vector<int>> & save,vector<int> temp){
      if(!root) return;
      temp.push_back(root->val); 
      if(!root->left and !root->right){
          if(targetSum-root->val==0) {save.push_back(temp);return;}
          else return;
      }
      dfs(root->left,targetSum-root->val,save,temp);
      dfs(root->right,targetSum-root->val,save,temp);
      
  }
```

# 链表

## 链表翻转(可用迭代和递归两种方法)

```cpp
class Solution {
  public:
      ListNode* reverseList(ListNode* head) {
          //迭代方法
          // if (!head ||!head->next) return head;
          // ListNode * re=head;
          // ListNode * f=head->next;
          // ListNode * s=f->next;
          // re->next=nullptr;
          // while(true){
          //     f->next=re;
          //     re=f;
          //     f=s;
          //     if(!f) break;
          //     s=s->next;
          // }
          // return re;
          //递归
          if (!head ||!head->next) return head;
          ListNode * f=head;
          ListNode * s=head->next;
          while(head->next) head=head->next;
          back_tracking(f,s);
          f->next=nullptr;
          return head;
      

      }
      void back_tracking(ListNode * f,ListNode * s){
          if(!s) {head1=f;return;}
          back_tracking(f->next,s->next);
          s->next=f;
      }
  };
```

## 链表翻转ii

```cpp
class Solution {
  public:
      ListNode* reverseBetween(ListNode* head, int m, int n) {
          if(!head) return head;
          ListNode* f=new ListNode();
          ListNode* head1=f;
          f->next=head;
          ListNode * end=head;
          for(int i=1;i<m;i++) f=f->next;
          for(int j=1;j<n;j++) end=end->next;
          ListNode * start=end->next;
          ListNode * new_f=f->next;
          ListNode * new_s=new_f->next;
          back_tracking(new_f,new_s,end);
          f->next=end;
          new_f->next=start;
          return head1->next;

      }
      void back_tracking(ListNode * f,ListNode* s,ListNode * end){
          if(f==end) return;
          back_tracking(f->next,s->next,end);
          s->next=f;
      }
  };
```

## 可以将每个节点存储于数组当中，这样方便进行下标操作。

```cpp
class Solution {
  public:
  void reorderList(ListNode* head) {
      //不用数组存储
      // ListNode * head1=head;
      // ListNode * node=head1;
      // if(!node ||! node->next) return ;
      // while(node){
      // while(node->next)node=node->next;
      // if(head1->next==node) return ;
      // node->next=head1->next;
      // head1->next=node;
      // ListNode *node1=node->next;
      // while(node1->next!=node) node1=node1->next;
      // node1->next=nullptr;
      // node=node->next;
      // head1=node;}

      //使用数组
      if(!head) return;
      vector<ListNode *> list;
      ListNode * node=head;
      while(node){
          list.push_back(node);
          node=node->next;
      }
      int i=0;int j=list.size()-1;
      while(i<j){
          if(list[i]->next==list[j]) break;
          list[j]->next=list[i]->next;
          list[i]->next=list[j];
          j--;
          i++;  
      }
      list[j]->next=nullptr;  
  }
  
  };
```

# Binary Search

## 例题1

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。经典的二分法搜索，如果这个值不存在于数组中，在某一次二分中，值一定会超过left和right组成的界限。

```cpp
int searchInsert(vector<int>& nums, int target) {
    int len=nums.size();
    if (len==0) return 0;
    if (len==1) return target>nums[0]; 
    int left=0;
    int right=len-1;
    while(left<=right){
        if (nums[left]>target) return left;
        if(nums[right]<target) return right+1;
        int mid=(left+right)/2;
        if (nums[mid]==target) return mid;
        else if(nums[mid]<target) left=mid+1;
        else right=mid-1;
    }
    return 0;
```

## 例题2

编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性。其实就是用两次二分。

```cpp
bool searchMatrix(vector<vector<int>>& matrix, int target) {
     int top=0;
     int down=matrix.size()-1;
     int len=matrix[0].size();
     int row;
     while(top<=down){
         if(matrix[top][len-1]>target) {row=top;break;}
         if(matrix[down][len-1]<target){row=down+1;break;}
         int mid=(top+down)/2;
         if(matrix[mid][len-1]<target) top++;
         else if (matrix[mid][len-1]>target) down--;
         else return true;
     }
     if (row>matrix.size()-1) return false;
     int left=0;int right=len-1;
     while(left<=right){
         if(matrix[row][left]>target) return false;
         if(matrix[row][right]<target) return false;
         int mid=(left+right)/2;
         if(matrix[row][mid]<target) left++;
         else if (matrix[row][mid]>target) right--;
         else return true;
     }
     return false;

 }
```
# 位运算

* &（与）｜（非）^(异或)
* 汉明运算：n&(n-1) 会将n的最右边的1改为0：主要用于判断一个数二进制中1的个数。（例201，231，191，338）
* $n_1$^ $n_2$ ^ $n_3...$ ^$n_n$,在一个数组中，若个一个数只出现了一次，其余数都出现了偶数次，则return这个数。（136，137，268）

# 图论

## 图的最短路径

主要分为三类：

* Dijkstra：没有负权边的单元最短路径
* Floyed：多元最短路径
* Bellman：含有负权边的单元最短路径

### Bellman-ford 算法

Bellman-ford 算法比dijkstra算法更具普遍性，因为它对边没有要求，可以处理负权边与负权回路。缺点是时间复杂度过高，高达O(VE), V为顶点数，E为边数。

其主要思想：对所有的边进行n-1轮松弛操作，因为在一个含有n个顶点的图中，任意两点之间的最短路径最多包含n-1边。换句话说，第1轮在对所有的边进行松弛后，得到的是源点最多经过一条边到达其他顶点的最短距离（已经是最优），（其他的经过多次条边达到的节点也会更新，只是不是最优）；第2轮在对所有的边进行松弛后，得到的是源点最多经过两条边到达其他顶点的最短距离；第3轮在对所有的边进行松弛后，得到的是源点最多经过一条边到达其他顶点的最短距离......

```cpp
for (var i = 0; i < n - 1; i++) {
    for (var j = 0; j < m; j++) {//对m条边进行循环
      var edge = edges[j];
      // 松弛操作
      if (distance[edge.to] > distance[edge.from] + edge.weight ){ 
        distance[edge.to] = distance[edge.from] + edge.weight;
      }
    }
}
```

例题：leetcode 787 K站中转内最便宜的航班

这里会限制最多经过K次中转，所以稍有不一样。第i次内循环，只对经过不大于i个中转站的节点更新

```cpp
class Solution2 {
public:
    int findCheapestPrice(int N, vector<vector<int>>& flights, int src, int dst, int K) {
        vector<int> dist(N, INT_MAX);
        // Initialize direct flight value
        dist[src] = 0;
        for (const auto& f : flights) {
            if (f[0] == src) {
                dist[f[1]] = f[2];
            }
        }

        for (int i = 0; i < K; i++) {
            auto dp = dist;
            for (const auto& f : flights) {
                auto u = f[0];
                auto v = f[1];
                auto w = f[2];
                if ((dist[u] != INT_MAX) && (dist[u] + w < dp[v])) {
                    dp[v] = dist[u] + w;
                }
            }
            dist = dp;
        }

        return dist[dst] == INT_MAX ? -1 : dist[dst];
    }
};
```

### Floyd 算法

通过Floyd计算图$G=(V,E)$ 中各个顶点的最短路径时，需要引入一个矩阵S，矩阵S中的元素$a[i][j]$ 表示顶点$i$ 到顶点$j$ 的距离。

假设图$G$ 中的顶点个数为$N$，则需要对矩阵进行$N$次更新。每次更新会将其中一个顶点作为中介点，然后去更新邻接矩阵所有的边。

```cpp
for (k = 0; k < mVexNum; k++)
    {
        for (i = 0; i < mVexNum; i++)
        {
            for (j = 0; j < mVexNum; j++)
            {
                // 如果经过下标为k顶点路径比原两点间路径更短，则更新dist[i][j]和path[i][j]
                tmp = (dist[i][k]==INF || dist[k][j]==INF) ? INF : (dist[i][k] + dist[k][j]);
                if (dist[i][j] > tmp)
                {
                    // "i到j最短路径"对应的值设，为更小的一个(即经过k)
                    dist[i][j] = tmp;
                    // "i到j最短路径"对应的路径，经过k
                    path[i][j] = path[i][k];
                }
            }
        }
    }
```

### Dijkstra算法

每次将离源点最近的点加入到集合中，然后用这个点为中介点去更新其他点的距离，然后再次找到最近的点加入到集合当中，按此循环，直到所有的点都加入到集合当中。目前，DIJ算法的复杂度是$O(n^2)$的，在一些题目中这个复杂度是不满足要求的。

#### 堆优化

pair是C++自带的二元组。我们可以把它理解成一个有两个元素的结构体。更刺激的是，这个二元组有自带的排序方式：以第一关键字为关键字，再以第二关键字为关键字进行排序。所以，我们用二元组的first位存距离，second位存编号即可。

* 优先队列是大根堆，需要把它变成小根堆
* 第一种是把第一关键字

```cpp
priority_queue<pair<int,int>>q;//最大堆，取反变为最小堆。
int nodes;//节点数
void dijkstra()
{
    vector<int> x(nodes,0);
    dist[1]=0;
    q.push(make_pair(0,1));
    while(!q.empty())
    {
        int x=q.top().second;
        q.pop();
        if(v[x])//说明该节点已经加入集合
            continue;
        v[x]=1;
        for(int i=head[x];i;i=nxt[i])
        {
            int y=to[i];
            if(dist[y]>dist[x]+val[i])
            {
                dist[y]=dist[x]+val[i];
                q.push(make_pair(-dist[y],y));
            }
        }
    }
}
```

## Prim

**普里姆算法**（prim），图论中的一种算法，可在加权连通图里搜索**最小生成树**。意即由此算法搜索到的边子集所构成的树中，不但包括了连通图里的所有顶点（英语：Vertex (graph theory)），且其所有边的权值之和亦为最小。

算法思路：从某个顶点开始，假设v0，此时v0属于最小生成树结点中的一个元素，该集合假设u，剩下的V-v0为待判定的点，此时选取u中的顶点到V-v0中顶点的一个路径最小的边，并且将其中非u中的顶点加入到u中，循环直到u中的顶点包含图所有的顶点为止。

```cpp
priority_queue<pair<int,int>>q;//最大堆，取反变为最大堆。
int nodes;//节点数
void prim()
{
    vector<int> x(nodes,0);
    dist[1]=0;
    q.push(make_pair(0,1));
    while(!q.empty())
    {
        int x=q.top().second;
        q.pop();
        if(v[x])//说明该节点已经加入集合
            continue;
        v[x]=1;
        for(int i=0;i<nodes;i++)
        {
          if(v[i]) continue;
          q.push(make_pair(-val[x][i],i));
            }
        }
    }
}
```

## 并查集

* 一般用于判断图是否为闭环
* 先设一个vector find，下标为该节点，值为该节点连通的节点。大小为图的节点的数目，初始值均为-1.
* 遍历每条边,取出每条边的节点：i，j。
* 用find[i]去找到i节点最终会连通的节点，找到值为-1的下`标，就是最终连通的节点。y同理。
* 如果发现两个值一样，说明这个图是连通的。否则将find[i]=y，更新连通节点。
```cpp
例题：冗余连接，684
class Solution {
public:
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int n=edges.size();
        vector<int> u(n+1,-1);
        for(int i=0;i<n;i++){
            int f=edges[i][0];
            int s=edges[i][1];
            while(u[f]!=-1||u[s]!=-1){
                if(u[f]!=-1) f=u[f];
                if(u[s]!=-1) s=u[s];
            }
            if(f==s) return edges[i];
            else u[f]=s;
        }
        return edges[0];
    }
};
```
## 拓扑排序

拓扑排序（Topological Sorting）是一个有向无环图（DAG, Directed Acyclic Graph）的所有顶点的线性序列。且该序列必须满足下面两个条件：
* 每个顶点出现且只出现一次。
* 若存在一条从顶点 A 到顶点 B 的路径，那么在序列中顶点 A 出现在顶点 B 的前面。
* ![avatar](top.png)
* 判断一个图是否为DAG的方式为：
    * 从 DAG 图中选择一个 没有前驱（即入度为0）的顶点并输出。
    * 从图中删除该顶点和所有以它为起点的有向边。
    * 重复 1 和 2 直到当前的 DAG 图为空或当前图中不存在无前驱的顶点为止。后一种情况说明有向图中必然存在环
    * ![avatar](top1.png)
```cpp
#include<iostream>
#include <list>
#include <queue>
using namespace std;

/************************类声明************************/
class Graph
{
    int V;             // 顶点个数
    list<int> *adj;    // 邻接表
    queue<int> q;      // 维护一个入度为0的顶点的集合
    int* indegree;     // 记录每个顶点的入度
public:
    Graph(int V);                   // 构造函数
    ~Graph();                       // 析构函数
    void addEdge(int v, int w);     // 添加边
    bool topological_sort();        // 拓扑排序
};

/************************类定义************************/
Graph::Graph(int V)
{
    this->V = V;
    adj = new list<int>[V];

    indegree = new int[V];  // 入度全部初始化为0
    for(int i=0; i<V; ++i)
        indegree[i] = 0;
}

Graph::~Graph()
{
    delete [] adj;
    delete [] indegree;
}

void Graph::addEdge(int v, int w)
{
    adj[v].push_back(w); 
    ++indegree[w];
}

bool Graph::topological_sort()
{
    for(int i=0; i<V; ++i)
        if(indegree[i] == 0)
            q.push(i);         // 将所有入度为0的顶点入队

    int count = 0;             // 计数，记录当前已经输出的顶点数 
    while(!q.empty())
    {
        int v = q.front();      // 从队列中取出一个顶点
        q.pop();

        cout << v << " ";      // 输出该顶点
        ++count;
        // 将所有v指向的顶点的入度减1，并将入度减为0的顶点入栈
        list<int>::iterator beg = adj[v].begin();
        for( ; beg!=adj[v].end(); ++beg)
            if(!(--indegree[*beg]))
                q.push(*beg);   // 若入度为0，则入栈
    }

    if(count < V)
        return false;           // 没有输出全部顶点，有向图中有回路
    else
        return true;            // 拓扑排序成功
}
```

# 单调栈

​    栈内元素是单调递增或递减的。例如 接雨水这道题，当出现准备入栈的元素的值是大于栈顶元素，那就开始出栈，直到栈顶元素大于入栈元素的值。

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        //单调栈
        if(height.size()==0||height.size()==1) return 0;
        stack<int> stk;
        stk.push(0);
        int ans=0;
        for(int i=1;i<height.size();i++){
            int top=stk.top();
            if(height[i]<=height[top]) stk.push(i);
            else{
                while(true){
                    top=stk.top();
                    int h=height[top];
                    stk.pop();
                    int j;
                    if(!stk.empty()) j=stk.top();
                    else break;
                    ans+=(i-j-1)*(min(height[j],height[i])-h);
                    if(height[i]<height[j]) break;

                }
                stk.push(i);

            }
        }
        return ans;
    }
};
```

例84

​    柱形图中最大矩形 84

思路：遍历以每个柱子为高形成的矩形的面积。这里可以用两个单调栈，一个为了找到该柱子的左边小于它高度的最小距离，另一个为了找到右边小于它高度的最小距离。

```cpp
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        stack<int> left;
        stack<int> right;
        vector<int> l;
        vector<int> r;
        if(heights.size()==1) return heights[0];
        left.push(0);
        l.push_back(1);
        for(int i=1;i<heights.size();i++){
            while(!left.empty()){
                    int temp = left.top();
                    if(heights[i]>heights[temp]){
                        l.push_back(i-temp);
                        left.push(i);
                        break;
                    }
                    else left.pop();
                }
                if(left.empty()){
                    left.push(i); l.push_back(i+1);
                }
            }
        right.push(heights.size()-1);
        r.push_back(1);
        for(int i=heights.size()-2;i>=0;i--){
            while(!right.empty()){
                    int temp = right.top();
                    if(heights[i]>heights[temp]){
                        r.push_back(temp-i);
                        right.push(i);
                        break;
                    }
                    else right.pop();
                }
                if(right.empty()){
                    right.push(i); r.push_back(heights.size()-i);
                }
            }
        int max_s=0;
        for(int i=0;i<heights.size();i++){
            int length=l[i]+r[heights.size()-i-1]-1;
            max_s=max(max_s,length*heights[i]);
        }
        return max_s;
        
        
    }
};
```

# 堆

STL 默认建立的是最大堆，想要最小堆必须在push_heap,pop_heap,make_heap每一个函数后面加第三个参数greater<int>()，括号不能省略。

* make_heap(_First, _Last, _Comp); //默认是建立最大堆的。对int类型，可以在第三个参数传入greater<int>()得到最小堆
* push_heap (_First, _Last); //在堆中添加数据,要先在容器中加入数据，再调用push_heap ()
* pop_heap(_First, _Last); //在堆中删除数据,要先调用pop_heap()再在容器中删除数据
* sort_heap(_First, _Last); //堆排序,排序之后就不再是一个合法的heap了

```cpp
#include <iostream>
#include <algorithm>
#include <vector>
 
using namespace std;
 
int main () {
  int a[] = {1,2,3,5,6};
  vector<int> v(a, a + sizeof(a)/sizeof(a[0]));
 
  make_heap(v.begin(),v.end());
  cout << "initial max of heap   : " << v.front() << endl;
 
  pop_heap(v.begin(),v.end()); v.pop_back();
  cout << "max of heap after pop : " << v.front() << endl;
 
  v.push_back(7); push_heap(v.begin(),v.end());
  cout << "max of heap after push: " << v.front() << endl;
 
  sort_heap (v.begin(),v.end());
 
  cout << "sorted range :";
  for (unsigned i=0; i<v.size(); i++) 
  		cout << " " << v[i];
  return 0;
}
```

topK

```cpp
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        make_heap(nums.begin(),nums.end());
        for(int i=0;i<k-1;i++){
            pop_heap(nums.begin(),nums.end());
            nums.pop_back();
        }
        return nums.front();
    }
};
```

# 排序

## 快排

时间复杂度$O(n^2)$

```cpp
#include<vector>
#include<iostream>
using namespace std;
void quicksort(vector<int>& nums, int start,int end){
    if(start>=end) return;
    int l=start;
    int r=end;
    int a=nums[start]; // 这里的a其实就是中间数，每一轮排序后a的左边小于a，a的右边大于a
    while(l<r){
        while(nums[r]>a and l<r) r--;
        while(nums[l]<=a and l<r) l++;
        if(l<r){
            int temp=nums[l];
            nums[l]=nums[r];
            nums[r]=temp;
        }
    }
    nums[start]=nums[l]; // 
    nums[l]=a;
    quicksort(nums,start,l);
    quicksort(nums,l+1,end);
}
```



## 选择排序

时间复杂度$O(n^2)$

具体思路：也要进行6次循环。第一次循环，将2与后面的六个数进行比较，选出最小的即1放在第一个位置；下一次则是从第二个数开始，与后面的五个数进行比较，然后将第二小的数即2放在第二个位置；然后以此类推，直到排完为止。 

```cpp
void choosesort(vector<int>& nums){
    for(int i=0;i<nums.size()-1;i++){
        for(int j=i+1;j<nums.size();j++){
            if(nums[i]>nums[j]){
                int temp=nums[i];
                nums[i]=nums[j];
                nums[j]=temp;
            }
        }
    }
```



## 插入排序 

时间复杂度$O(n^2)$

1. 从第一个元素开始，该元素可以认为已经被排序
2. 取出下一个元素，在已经排序的元素序列中从后向前扫描
3. 如果该元素（已排序）大于新元素，将该元素移到下一位置
4. 重复步骤3，直到找到已排序的元素小于或者等于新元素的位置
5. 将新元素插入到该位置后
6. 重复步骤2~5

```cpp
void insertSort(vector<int> &array){
	//从第二个元素开始，加入第一个元素是已排序数组
	int N=array.size();
    for (int i = 1; i < N; i++){
		//待插入元素 array[i]
		if (array[i] < array[i - 1]){
			int wait = array[i];
			int j = i;
			while (j > 0 && array[j - 1] > wait){
				//从后往前遍历已排序数组，若待插入元素小于遍历的元素，则遍历元素向后挪位置
				array[j] = array[j - 1];
				j--;
			}
			array[j] = wait;
		}
	}
}
```



## 归并排序

时间复杂度$O(nlogn)$

```cpp
vector<int> mergesort(vector<int> & nums, int start,int end){
    if(start==end) return {nums[start]};
    int mid=(start+end)/2;
    vector<int> l=mergesort(nums,start,mid);
    vector<int> r=mergesort(nums,mid+1,end);
    vector<int> combine(end-start+1);
    int i=0;
    int j=0;
    int k=0;
    while(i<l.size() || j<r.size()){
        if(i<l.size() && j<r.size()){
            if(l[i]<=r[j]){
                combine[k]=l[i];
                i++;}
            else {
                combine[k]=r[j];
                j++;}

        }
        else if(i<l.size()){
            combine[k]=l[i];
            i++;
        }
        else{
           combine[k]=r[j];
            j++; 
        }
        k++;
    }
    return combine;

}
```

