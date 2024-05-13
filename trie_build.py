import numpy as np
import math
import random
import matplotlib.pyplot as plt

def print_val(val: int) -> None:
    print(val * np.pi)
    print(math.sqrt(val))

def find_smallest_missing_integer(A):
        """
        Given an array A of N integers, returns the smallest positive integer 
        (greater than 0) that does not occur in A.
        
        Input: A is a list of N integers (1 <= N <= 100,000), where each 
        integer is between -1,000,000 and 1,000,000.
        Output: An integer (1 <= result <= 1,000,000)
        
        Time complexity: O(N)"""

        # Create a set to store the unique elements in A
        unique_elements = set(A)

        # Start from 1 and check if each integer is present in the set
        smallest_missing = 1
        while smallest_missing in unique_elements:
            smallest_missing += 1

        return smallest_missing



def test_find_smallest_missing_integer():
    # Test case 1: A contains positive integers
    A = [1, 3, 6, 4, 1, 2]
    expected_output = 5
    assert find_smallest_missing_integer(A) == expected_output

    # Test case 2: A contains negative integers
    A = [-1, -3, -6, -4, -1, -2]
    expected_output = 1
    assert find_smallest_missing_integer(A) == expected_output

    # Test case 3: A contains both positive and negative integers
    A = [-1, 3, -6, 4, -1, 2]
    expected_output = 1
    assert find_smallest_missing_integer(A) == expected_output

    # Test case 4: A contains only negative integers
    A = [-1, -3, -6, -4, -5, -2]
    expected_output = 1
    assert find_smallest_missing_integer(A) == expected_output

    # Test case 5: A contains only positive integers
    A = [1, 2, 3, 4, 5, 6]
    expected_output = 7
    assert find_smallest_missing_integer(A) == expected_output

    # Test case 6: A contains duplicate positive integers
    A = [1, 1, 1, 1, 1]
    expected_output = 2
    assert find_smallest_missing_integer(A) == expected_output

    # Test case 7: A contains duplicate negative integers
    A = [-1, -1, -1, -1, -1]
    expected_output = 1
    assert find_smallest_missing_integer(A) == expected_output

    # Test case 8: A contains a mix of positive, negative, and zero
    A = [-1, 0, 1, 2, 3]
    expected_output = 4
    assert find_smallest_missing_integer(A) == expected_output

    print("All test cases passed!")


def wordBreak(s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        # Create a set of words in the dictionary
        word_set = set(wordDict)
        
        # Create a list to store the results of subproblems
        dp = [False] * (len(s) + 1)
        
        # Base case: empty string can be broken
        dp[0] = True
        
        # Iterate over the string
        for i in range(1, len(s) + 1):
            for j in range(i):
                # Check if the substring s[j:i] can be broken
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        
        return dp[len(s)]

def lengthOfLIS(nums):
            """
            Given a list of integers nums, returns the length of the longest increasing subsequence.

            Input: nums is a list of integers.
            Output: An integer representing the length of the longest increasing subsequence.

            Time complexity: O(nlogn)
            """

            # Create a list to store the increasing subsequence
            lis = []

            # Iterate over the numbers in nums
            for num in nums:
                # Use binary search to find the position to insert num in lis
                left, right = 0, len(lis) - 1
                while left <= right:
                    mid = (left + right) // 2
                    if lis[mid] < num:
                        left = mid + 1
                    else:
                        right = mid - 1

                # If num is greater than all elements in lis, append it to the end
                if left == len(lis):
                    lis.append(num)
                # Otherwise, replace the element at left with num
                else:
                    lis[left] = num

            # Return the length of the longest increasing subsequence
            return len(lis)

def Unique(s):
    if len(s) <= 1: return True
    for i in range(1,len(s)):
          if s[i] == s[i-1]:
               return False
    return True

def CheckPermutations(s1:str, s2:str) -> bool:
    if len(s1) != len(s2):
          return False
    apperences = [0] * 128
    for i in range(len(s1)):
          apperences[ord(s1[i]) - ord('a')] += 1
          apperences[ord(s2[i]) - ord('a')] -= 1
    return True if apperences == [0] * 128 else False

def URlify(s:str) -> str:

    gaps = 0
    for i in range(len(s)):
         if s[i] == " ":
              gaps += 1

    s_list = list(s) + [None] * gaps * 2

    i = len(s_list) - 1
    j = len(s_list) - 1

    while i > 0:
        if s_list[i] != " " and s_list[i] != None:
            s_list[j] = s_list[i]
            j -= 1
        elif s_list[i] == " ":
            s_list[j] = "0"
            s_list[j - 1] = "2"
            s_list[j - 2] = "%"
            j -= 3
        i -= 1

    s = "".join(s_list)
    return s

def PalindromPermutations(s: str) -> bool:
    repetitions = [0] * 26
    letters = 0
    for c in s:
        if ord("a") <= ord(c) <= ord("z"):
            repetitions[ord(c) - ord("a")] += 1
            letters += 1
        elif ord("A") <= ord(c) <= ord("Z"):
            repetitions[ord(c) - ord("A")] += 1
            letters += 1
    num_of_uneven = sum([val % 2 for val in repetitions])
    return True if num_of_uneven == letters % 2 else False

def OneWay(s1: str, s2: str) -> bool:
    if len(s1) == len(s2):
        return OneWayReplace(s1,s2)
    elif len(s1) == len(s2) + 1:
        return OneWayInsert(s1,s2)
    elif len(s1) == len(s2) - 1:
        return OneWayInsert(s2,s1)
    else:
        return False
    
def OneWayReplace(s1: str, s2: str) -> bool:
     num_of_errors = 0
     for i in range(len(s1)):
          if s1[i] != s2[i]:
               num_of_errors += 1
               if num_of_errors >= 2:
                    return False
     return True

def OneWayInsert(s1: str, s2: str) -> bool:
    i1 = 0
    i2 = 0
    while i1 < len(s1) and i2 < len(s2):
         if s1[i1] != s2[i2]:
              if i1 == i2:
                   i1 += 1
              else:
                   return False
         else:
              i1 += 1
              i2 += 1
    return True

def StringCompression(s: str) -> str:
    new_s = ""
    count = 0
    for i in range(len(s)):
        if count == 0:
            new_s += s[i]
            count += 1
        if i < len(s) - 1 and s[i] != s[i + 1]:
            if count >= 2:
                new_s += str(count)
            count = 0
        if i < len(s) - 1 and s[i] == s[i + 1]:
            count += 1
    if count >= 2:
        new_s += str(count)
    return new_s

def RotateMatrix(mat: list[list[int]]) -> list[list[int]]:
    n = len(mat)
    init = 0
    final = n - 1
    while init < final:
        for i in range(init, final):
            temp = [mat[init][i], mat[i][n - 1 - init], \
                    mat[n - 1 - init][n - 1 - i], \
                    mat[n - 1 - i][init]]
            mat[init][i] = temp[3]
            mat[i][n - 1 - init] = temp[0]
            mat[n - 1 - init][n - 1 - i] = temp[1]
            mat[n - 1 - i][init] = temp[2]
        init += 1
        final -= 1
    return mat

def ZeroMatrix(mat: list[list[int]]) -> list[list[int]]:
    if len(mat) == 0 or len(mat[0]) == 0:
        return mat
    zero_rows = [0] * len(mat)
    zero_cols = [0] * len(mat[0])
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 0:
                zero_rows[i] = 1
                zero_cols[j] = 1
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if zero_rows[i] == 1 or zero_cols[j] == 1:
                mat[i][j] = 0
    return mat

class Node:
    def __init__(self, data):
        self.val = data
        self.next = None

def RemoveDups(head: Node) -> Node:
    if not head:
        return head
    dict = {head.val: 1}
    node = head.next
    prev_node = head
    while node:
        if node.val not in dict:
            dict[node.val] = 1
            prev_node = node
            node = node.next
        else:
            prev_node.next = node.next
            node = node.next
    return head

def Partition(head: Node, x: int) -> Node:
    if not head or not head.next:
        return head
    node_large = head
    node_small = head
    steps_large = 0
    steps_small = 0
    while node_large and node_small:
        while node_large.val < x:
            node_large = node_large.next
            steps_large += 1
            if not node_large:
                return head
            while node_small.val >= x or steps_small < steps_large:
                node_small = node_small.next
                steps_small += 1
                if not node_small:
                    return head
            node_large.val, node_small.val = node_small.val, node_large.val
    return head

def sumLists(head1: Node, head2: Node) -> Node:
    extra_val = 0
    new_head = None
    prev_node = new_head
    while head1 or head2:
        if head1:
            val1 = head1.val
            head1 = head1.next
        else:
            val1 = 0
        if head2:
            val2 = head2.val
            head2 = head2.next
        else:
            val2 = 0
        if val1 + val2 + extra_val < 10:
            new_val = val1 + val2 + extra_val
            extra_val = 0
        else:
            new_val = val1 + val2 + extra_val - 10
            extra_val = 1
        new_node = Node(new_val)
        if prev_node:
            prev_node.next = new_node
            prev_node = new_node
        else:
            new_head = new_node
            prev_node = new_node
    return new_head

def numFromList(head: Node) -> int:
    num = 0
    while head:
        num = num * 10 + head.val
        head = head.next
    return num

def sumLists2(head1: Node, head2: Node) -> Node:
    num1 = numFromList(head1)
    num2 = numFromList(head2)
    num3 = num1 + num2
    prev_node = None
    while num3 > 0:
        new_node = Node(num3 % 10)
        new_node.next = prev_node
        prev_node = new_node
        num3 = num3 // 10
    return new_node

def reverseLinkedList(head: Node) -> Node:
    head2 = Node(head.val)
    head = head.next
    while head:
        new_node = Node(head.val)
        new_node.next = head2
        head2 = new_node
        head = head.next
    return head2

def palindromLinkedList(head: Node) -> bool:
    if not head or not head.next:
        return True
    head2 = reverseLinkedList(head)
    while head:
        if head.val != head2.val:
            return False
        head = head.next
        head2 = head2.next
    return True

def intersection(head1: Node, head2: Node) -> bool:
    if (head1 and not head2) or (head2 and not head1):
        return False
    node1 = head1
    node2 = head2
    node1_in_list2 = False
    node2_in_list1 = False
    while not(node1_in_list2) and not(node2_in_list1) \
        and node1 and node2:
        if node1 == node2:
            return True
        if node1.next == None:
            node1_in_list2 = True
            node1 = head2
        else:
            node1 = node1.next
        if node2.next == None:
            node2_in_list1 = True
            node2 = head1
        else:
            node2 = node2.next
    return False

def loopDetection(head: Node) -> Node:
    if not head or not head.next:
        return None
    node_slow = head.next
    node_fast = head.next.next
    while node_slow != node_fast:
        if not node_fast or not node_fast or not node_fast.next:
            return None
        node_slow = node_slow.next
        node_fast = node_fast.next.next
    while node_slow != head:
        node_slow = node_slow.next
        head = head.next
    return node_slow

class threeInOne:
    def __init__(self) -> None:
        self.array = [0] * 12
        self.stacks_i = [-3, -2, -1]

    def push(self, val: int, stack_num: int) -> None:
        self.stacks_i[stack_num] += 3
        if self.stacks_i[stack_num] >= len(self.array):
            self.array = self.array + [0] * len(self.array)
        self.array[self.stacks_i[stack_num]] = val

    def isEmpty(self, stack_num: int) -> bool:
        return self.stacks_i[stack_num] < 0 
    
    def peek(self, stack_num: int) -> int:
        if self.isEmpty(stack_num):
            return None
        return self.array[self.stacks_i[stack_num]]
    
    def pop(self, stack_num: int) -> int:
        if self.isEmpty(stack_num):
            return None
        val = self.array[self.stacks_i[stack_num]]
        self.stacks_i[stack_num] -= 3
        if max(self.stacks_i) < len(self.array) / 4 and len(self.array) > 12:
            self.array = self.array[:len(self.array) / 2]
        return val
    
class StackMin:
    def __init__(self) -> None:
        self.stack = None
        self.min_stack = None

    def push(self, val: int) -> None:
        new_node = Node(val)
        new_node.next = self.stack
        self.stack = new_node
        if not self.min_stack or self.min_stack.val >= val:
            new_min_node = Node(val)
            new_min_node.next = self.min_stack
            self.min_stack = new_min_node
    
    def pop(self) -> int:
        if not self.stack:
            return None
        if self.min_stack.val == self.stack.val:
            self.min_stack = self.min_stack.next
        val = self.stack.val
        self.stack = self.stack.next
        return val
    
    def get_min(self) -> int:
        return self.min_stack.val

class StackNode:
    def __init__(self,val: int) -> None:
        self.val = val
        self.head_min_stack = None
        self.next_big_node = None

class StackOfPlates:
    def __init__(self,max_in_stack: int) -> None:
        self.max_vals = max_in_stack
        self.head = StackNode(max_in_stack)
    
    def push(self, val: int) -> None:
        if self.head.val == 0:
            new_big_node = StackNode(self.max_vals)
            new_big_node.next_big_node = self.head
            self.head = new_big_node

        self.head.val -= 1
        new_node = Node(val)
        new_node.next = self.head.head_min_stack
        self.head.head_min_stack = new_node

    def pop(self) -> int:
        if not self.head.head_min_stack:
            return None
        val = self.head.head_min_stack.val
        self.head.val += 1
        if self.head.val == self.max_vals:
            self.head = self.head.next_big_node
        else:
            self.head.head_min_stack = self.head.head_min_stack.next
        return val
    
    def popAt(self, index: int) -> int:
        if index == 0 or not self.head.head_min_stack:
            return self.pop()
        prev_big_node = self.head
        curr_big_node = self.head.next_big_node
        if not curr_big_node:
            return None
        for i in range(index - 1):
            prev_big_node = curr_big_node
            curr_big_node = curr_big_node.next_big_node
            if not curr_big_node:
                return None
        val = curr_big_node.head_min_stack.val
        curr_big_node.head_min_stack = curr_big_node.head_min_stack.next
        curr_big_node.val += 1
        if curr_big_node.val == self.max_vals:
            prev_big_node.next_big_node = curr_big_node.next_big_node
        return val

class MyStack:
    def __init__(self) -> None:
        self.stack = None

    def push(self, val: int) -> None:
        new_node = Node(val)
        new_node.next = self.stack
        self.stack = new_node

    def pop(self) -> int:
        if not self.stack:
            return None
        val = self.stack.val
        self.stack = self.stack.next
        return val
    
    def peek(self) -> int:
        if not self.stack:
            return None
        return self.stack.val
    
    def isEmpty(self) -> bool:
        return not self.stack
    

    
def copyStack(stack: MyStack) -> MyStack:
    new_stack = MyStack()
    queue = MyQueue()

    while not stack.isEmpty():
        val = stack.pop()
        queue.append(val)
        new_stack.push(val)

    while not queue.isEmpty():
        stack.push(queue.pop())

    return new_stack

def sortStack(stack: MyStack) -> MyStack:
    stack2 = MyStack()
    while not stack.isEmpty():
        temp = stack.pop()
        while not stack2.isEmpty() and stack2.peek() > temp:
            stack.push(stack2.pop())
        stack2.push(temp)
    while not stack2.isEmpty():
        stack.push(stack2.pop())
    return stack

class AnimalShelter:
    def __init__(self) -> None:
        self.dict = {"cat": 0, "dog": 1}
        self.heads = [None, None]
        self.tails = [None, None]
        self.count = 0

    def isEmpty(self, animal: str) -> bool:
        return not self.heads[self.dict[animal]]
    
    def push(self, animal: str) -> None:
        new_node = Node(self.count)
        self.count += 1
        if not self.heads[self.dict[animal]]:
            self.heads[self.dict[animal]] = new_node
            self.tails[self.dict[animal]] = new_node
        else:
            self.tails[self.dict[animal]].next = new_node
            self.tails[self.dict[animal]] = new_node
        
    def popAny(self) -> str:
        if self.isEmpty("cat") and self.isEmpty("dog"):
            return None
        elif self.isEmpty("cat") or \
            self.heads[self.dict["cat"]].val < self.heads[self.dict["dog"]].val:
            return self.popCat()
        else:
            return self.popDog()
        
    def popCat(self) -> str:
        return self.popSpecific("cat")
    
    def popDog(self) -> str:
        return self.popSpecific("dog")
    
    def popSpecific(self, animal: str) -> str:
        if self.heads[self.dict[animal]] == self.tails[self.dict[animal]]:
            self.heads[self.dict[animal]] = None
            self.tails[self.dict[animal]] = None
        else:
            self.heads[self.dict[animal]] = self.heads[self.dict[animal]].next
        return animal
    
class BFSNode:
    def __init__(self, val: int) -> None:
        self.val = val
        self.children = []

def routeBetweenNodes(node1: BFSNode, node2: BFSNode) -> bool:
    visited = [node1]
    queue = [node1]
    while len(queue) > 0:
        node = queue.pop(0)
        if node == node2:
            return True
        for child in node.children:
            if not child in visited:
                visited.append(child)
                queue.append(child)
    return False

class TreeNode:
    def __init__(self, val: int, left = None, right = None) -> None:
        self.val = val
        self.left = left
        self.right = right

def minimalTree(arr: list[int]) -> TreeNode:
    location = len(arr) // 2
    val = arr[location]
    if len(arr) == 1:
        return TreeNode(val)
    elif len(arr) == 2:
        return TreeNode(val, TreeNode(arr[0]))
    elif len(arr) >= 3:
        return TreeNode(val, minimalTree(arr[0:location]), minimalTree(arr[location + 1:]))


class MyQueue:
    def __init__(self) -> None:
        self.head = None
        self.tail = None

    def append(self, val) -> None:
        new_node = Node(val)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node

    def pop(self):
        if not self.head:
            return None
        val = self.head.val
        self.head = self.head.next
        return val
    
    def isEmpty(self) -> bool:
        return not self.head
    
    def queueToLinedList(self) -> Node:
        head = self.head
        new_head = Node(head.val.val)
        new_node = new_head
        head = head.next
        while head:
            new_node.next = Node(head.val.val)
            new_node = new_node.next
            head = head.next
        return new_head


def listOfDepth(root: TreeNode) -> list[Node]:
    res = [Node(root.val)]
    curr_queue = MyQueue()
    next_queue = MyQueue()
    curr_queue.append(root)
    while not curr_queue.isEmpty():
        curr_node = curr_queue.pop()
        if curr_node.left:
            next_queue.append(curr_node.left)
        if curr_node.right:
            next_queue.append(curr_node.right)
        if curr_queue.isEmpty() and not next_queue.isEmpty():
            new_head = next_queue.queueToLinedList()
            res.append(new_head)
            curr_queue = next_queue
            next_queue = MyQueue()
    return res

def checkBalanced(root: TreeNode) -> bool:
    if not root:
        return True
    return checkBalancedRecursive(root) [0]

def checkBalancedRecursive(node: TreeNode) -> tuple[bool, int]:
    if not node:
        return True, -1
    left_balance, left_height = checkBalancedRecursive(node.left)
    right_balance, right_height = checkBalancedRecursive(node.right)
    if left_balance and right_balance and \
        abs(left_height - right_height) <= 1:
        balance = True
    else:
        balance = False
    height = max(left_height, right_height) + 1
    return balance, height


class Res:
    def __init__(self) -> None:
        self.res = True
        self.prev_val = float("-inf")

    def inorder(self, node: TreeNode) -> bool:
        if node.left:
            left_bool = self.inorder(node.left)
            if left_bool == False:
                return False
        
        if node.val < self.prev_val:
            self.res = False
            return False
        self.prev_val = node.val

        if node.right:
            self.inorder(node.right)

def validateBST(root: TreeNode) -> bool:
    if not root:
        return True

    res = Res()
    res.inorder(root)
    return res.res

class GraphNode:
    def __init__(self, val = None) -> None:
        self.val = val
        self.children = MyQueue()
        self.num_of_parents = 0

    def add_child(self, child) -> None:
        self.children.append(child)
        child.num_of_parents += 1

    def remove_child(self):
        child = self.children.pop()
        child.num_of_parents -= 1
        return child

def nodesAndEdgesToGraph(nodes: list[str], edges: list[list[str]]) -> list[GraphNode]:
    graph_nodes = [None] * len(nodes)

    for i, node in enumerate(nodes):
        graph_nodes[i] = GraphNode(node)

    for edge in edges:
        parent = graph_nodes[nodes.index(edge[0])]
        child = graph_nodes[nodes.index(edge[1])]
        parent.add_child(child)
    return graph_nodes

def buildOrder(nodes: list[str], edges: list[list[str]]) -> list[str]:
    graph_nodes = nodesAndEdgesToGraph(nodes, edges)
    build_order = [None] * len(nodes)
    i = 0
    for node in graph_nodes:
        if node.num_of_parents == 0:
            build_order[i] = node.val
            i += 1

    j = 0
    while build_order[j]:
        curr_node = graph_nodes[nodes.index(build_order[j])]
        while not curr_node.children.isEmpty():
            child = curr_node.remove_child()
            if child.num_of_parents == 0:
                build_order[i] = child.val
                i += 1
        if i == len(nodes):
            return build_order
        j += 1
    return None

def firstCommonAncestor(root: TreeNode, node1: TreeNode, node2: TreeNode) -> TreeNode:
    return firstCommonAncestorRecursive(root, node1, node2)[0]

def firstCommonAncestorRecursive(node: TreeNode, node1: TreeNode, node2: TreeNode) -> tuple[TreeNode, bool, bool]:
    if not node:
        return None, False, False
    
    node_left, decendent1_left, decendent2_left = firstCommonAncestorRecursive(node.left, node1, node2)
    if node_left:
        return node_left, True, True
    
    node_right, decendent1_right, decendent2_right = firstCommonAncestorRecursive(node.right, node1, node2)
    if node_right:
        return node_right, True, True
    
    decendent1 = (node == node1 or decendent1_left or decendent1_right)
    decendent2 = (node == node2 or decendent2_left or decendent2_right)
    if decendent1 and decendent2:
        return node, True, True
    else:
        return None, decendent1, decendent2


def weaveLists(prefix: list, first: list, second: list, results: list) -> None:
    if len(first) == 0 or len(second) == 0:
        result = prefix + first + second
        results.append(result)
        return

    weaveLists(prefix + [first[0]], first[1:], second, results)
    weaveLists(prefix + [second[0]], first, second[1:], results)


def allSequences(root: TreeNode) -> list[list[int]]:
    if not root:
        return [[]]
        
    left_sequences = allSequences(root.left)
    right_sequences = allSequences(root.right)

    results = []
    for left in left_sequences:
        for right in right_sequences:
            weaved = []
            weaveLists([root.val], left, right, weaved)
            results += weaved

    return results

def checkSubtree(root1: TreeNode, root2: TreeNode) -> bool:
    if not root1 or not root2:
        return False
    queue = MyQueue()
    queue.append(root1)
    while not queue.isEmpty():
        node = queue.pop()
        if node.val == root2.val:
            if checkEqualTrees(node, root2):
                return True
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return False

def checkEqualTrees(node1: TreeNode, node2: TreeNode) -> bool:
    if not node1 and not node2:
        return True
    if not node1 or not node2:
        return False
    if node1.val != node2.val:
        return False
    return checkEqualTrees(node1.left, node2.left) and checkEqualTrees(node1.right, node2.right)

class BinaryTree:
    def __init__(self) -> None:
        self.root = None
        self.size = 0

    def insert(self, val: int) -> None:
        if not self.root:
            self.root = TreeNode(val)
            self.size += 1
        else:
            self.insertRecursive(self.root, val)

    def insertRecursive(self, node: TreeNode, val: int) -> None:
        if val < node.val:
            if not node.left:
                node.left = TreeNode(val)
                self.size += 1
            else:
                self.insertRecursive(node.left, val)
        else:
            if not node.right:
                node.right = TreeNode(val)
                self.size += 1
            else:
                self.insertRecursive(node.right, val)

    def find(self, val: int) -> bool:
        return self.findRecursive(self.root, val)
    
    def findRecursive(self, node: TreeNode, val: int) -> TreeNode:
        if not node:
            return None
        if node.val == val:
            return node
        elif val < node.val:
            return self.findRecursive(node.left, val)
        else:
            return self.findRecursive(node.right, val)
        
    def randomNode(self) -> TreeNode:
        if not self.root:
            return None
        random_val = random.randint(0, self.size - 1)
        return self.randomNodeRecursive(self.root, random_val)

def countPathsWithSum(root: TreeNode, target: int) -> int:
    dict = {}
    curr_sum = 0
    return recursiveCountPathsWithSum(root, target, curr_sum, dict)

def recursiveCountPathsWithSum(node: TreeNode, target: int, curr_sum: int, dict: dict) -> int:
    if not node:
        return 0
    
    new_sum = curr_sum + node.val
    counter = 1 if new_sum == target else 0
    if new_sum - target in dict:
        counter += dict[new_sum - target]
    
    if new_sum in dict:
        dict[new_sum] += 1
    else:
        dict[new_sum] = 1
    
    counter += recursiveCountPathsWithSum(node.left, target, new_sum, dict)
    counter += recursiveCountPathsWithSum(node.right, target, new_sum, dict)
    dict[new_sum] -= 1
    return counter

def getBit(num: int, i: int) -> bool:
    val = (1 << i)
    return (num & val) != 0

def isPrime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
    

def girlsToBoysRatio(n: int = 1000) -> None:
    ratio = [0] * n
    girls = 0
    boys = 0
    for i in range(0, n):
        while random.choice([True, False]):
            boys += 1
        girls += 1
        ratio[i] = girls / (girls + boys)
    plt.plot(ratio)
    plt.show()

def poison() -> None:
    bottles = 1000
    tests = 10
    poison = random.randint(0, bottles - 1)
    res = 0
    for i in range(0, tests):
        if getBit(poison, i):
            print("Test", i, "is positive")
            res += 2 ** i
        else:
            print("Test", i, "is negative")
    print("The poison bottle is", res)

class Card:
    def __init__(self, shape: str, color: str, value: int) -> None:
        self.shape = shape
        self.color = color
        self.value = value

class Deck:
    def __init__(self, deck_type: str = "regular") -> None:
        self.cards = []
        if deck_type == "regular":
            self.initializeRegularDeck()

    def initializeRegularDeck(self) -> None:
        for shape in ["hearts", "diamonds", "clubs", "spades"]:
            for color in ["red", "black"]:
                for value in range(1, 14):
                    self.cards.append(Card(shape, color, value))

    def shuffle(self) -> None:
        random.shuffle(self.cards)
    
    def draw(self) -> Card:
        return self.cards.pop()
    
    def isEmpty(self) -> bool:
        return len(self.cards) == 0
    
    def initializeDeck(self) -> None:
        self.cards = []
        self.initializeRegularDeck()
    
def blackJack() -> None:
    deck = Deck()
    deck.shuffle()
    player = 0
    house = 0
    while player < 21:
        card = deck.draw()
        player += card.value
        print("Playr drew", card.value, " of ", card.shape ,"and now has", player)
    while house < 17:
        card = deck.draw()
        house += card.value
        print("House drew", card.value, " of ", card.shape ,"and now has", house)
    if player > 21:
        print("Player lost")
    elif house > 21:
        print("Player won")
    elif player > house:
        print("Player won")
    else:
        print("Player lost")

def tripleStep(n: int) -> int:
    # Checking for edga cases
    if n == 0:
        return 1
    elif n == 1:
        return 1
    elif n == 2:
        return 2
    
    # Initializing data
    minus3 = 1
    minus2 = 1
    minus1 = 2
    
    for i in range(3, n + 1):
        val = minus3 + minus2 + minus1
        minus3 = minus2
        minus2 = minus1
        minus1 = val

    return val

def robotInGrid(r: int, c: int, off_limits: list[list[int]]) -> list[list[int]]:
    bool_grid = genorateBoolGrid(r, c, off_limits)
    res = [[0,0]]

    while res[-1] != [r - 1, c - 1]:
        if bool_grid[res[-1][0] + 1][res[-1][1]]:
            res.append([res[-1][0] + 1, res[-1][1]])
        elif bool_grid[res[-1][0]][res[-1][1] + 1]:
            res.append([res[-1][0], res[-1][1] + 1])
        else:
            return None
                       
    return res

def genorateBoolGrid(r: int, c: int, off_limits: list[list[int]]) -> list[list[bool]]:
    bool_grid = [[False] * (c + 1) for _ in range(r + 1)]
    bool_grid[r - 1][c - 1] = True
    off_limits_dict = dict()
    for off_limit in off_limits:
        off_limits_dict[(off_limit[0], off_limit[1])] = 1

    for i in range(r - 1, -1, -1):
        for j in range(c - 1, -1, -1):
            if ((i, j) not in off_limits_dict) and (bool_grid[i + 1][j] or bool_grid[i][j + 1]):
                bool_grid[i][j] = True
    
    return bool_grid

def robotInGrid2(grid: list[list[bool]]) -> list[list[int]]:
    grid_copy = grid.copy()
    r = len(grid_copy)
    c = len(grid_copy[0])
    
    for i in range(r - 1, -1 , -1):
        for j in range(c - 1, -1, -1):
            if not grid_copy[i][j] or \
                ((i < r - 1 and not grid_copy[i + 1][j]) and \
                (j < c - 1 and not grid_copy[i][j + 1])):
                grid_copy[i][j] = False
            else:
                grid_copy[i][j] = True

    res = [[0,0]]
    while res[-1] != [r - 1, c - 1]:
        curr_r = res[-1][0]
        curr_c = res[-1][1]
        if curr_r < r - 1 and grid_copy[curr_r + 1][curr_c]:
            res.append([curr_r + 1, curr_c])
        elif curr_c < c - 1 and grid_copy[curr_r][curr_c + 1]:
            res.append([curr_r, curr_c + 1])
        else:
            return None
    return res

def magicIndex(arr: list[int]) -> int:
    return magicIndexRecursive(arr, 0)

def magicIndexRecursive(arr: list[int], index_jump: int) -> int:
    i = round((len(arr) - 1)/ 2)

    if arr[i] == i + index_jump:
        return arr[i]
    
    if i == 0 or i == len(arr) - 1:
        return None
    
    elif arr[i] > i + index_jump:
        return magicIndexRecursive(arr[:i], index_jump)
        
    else:
        return magicIndexRecursive(arr[i + 1:], index_jump + i + 1)
    
def powerSet(set: list[int]) -> list[list[int]]:
    if len(set) == 0:
        return [[]]
    
    val = set[0]
    subsets = powerSet(set[1:])
    temp = []
    for subset in subsets:
        temp.append(subset + [val])

    return subsets + temp    

def multiplier(num1: int, num2: int) -> int:
    if num1 > num2:
        return recursiveMultiplier(num1, num2)
    else:
        return recursiveMultiplier(num2, num1)
    
def recursiveMultiplier(num1: int, num2: int) -> int:
    if num2 == 0:
        return 0
    if (num2 & 1) == 0:
        return recursiveMultiplier(num1 << 1, num2 >> 1)
    else:
        return num1 + recursiveMultiplier(num1, num2 - 1)
    

def HanoiTowersRecursion(init: MyStack, buffer: MyStack, goal: MyStack, size: int) -> None:
    # base case
    if size == 0:
        return
    
    # move the top n-1 values to buffer using the goal as the buffer
    HanoiTowersRecursion(init, goal, buffer, size - 1)

    # move the bottom value to the goal
    goal.push(init.pop())

    # move the buffer to the goal using initial as the buffer
    HanoiTowersRecursion(buffer, init, goal, size - 1)




if __name__ == "__main__":
    init = MyStack()
    goal = MyStack()
    buffer = MyStack()
    for i in range(4, 0, -1):
        init.push(i)
    HanoiTowersRecursion(init, buffer, goal, 4)
    while not goal.isEmpty():
        print(goal.pop())