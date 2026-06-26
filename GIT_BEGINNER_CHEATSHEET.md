# Git 新手速查表

这份文档面向第一次接触 Git 的同学，结合当前项目给出最常用的命令和典型场景。

当前项目目录：

```powershell
D:\project\project\mindpal_demo\agent_version
```

当前项目默认分支：

```powershell
main
```

当前项目远程仓库：

```powershell
origin -> git@github.com:Salley-Xu/mindpal-demo-agent.git
```

---

## 1. Git 是什么

Git 可以理解成代码的版本管理工具，也就是代码的“时间机器”。

你可以用它来做这些事：

- 记录每一次代码修改
- 查看某次改了什么
- 回到之前的版本
- 把本地代码推到 GitHub
- 和别人协作开发

---

## 2. 先记住 4 个概念

### 工作区

就是你当前正在编辑的文件。

### 暂存区

就是你准备提交的文件集合。

### 提交

也就是 `commit`，可以理解成一次正式保存。

### 远程仓库

也就是 GitHub 上的仓库，比如：

```powershell
git@github.com:Salley-Xu/mindpal-demo-agent.git
```

---

## 3. 最常用的日常流程

以后你大多数时候只要记住这 4 步：

```powershell
git status
git add .
git commit -m "写明这次改了什么"
git push
```

含义分别是：

1. `git status`
   查看当前哪些文件改了
2. `git add .`
   把当前改动加入暂存区
3. `git commit -m "说明"`
   生成一个版本快照
4. `git push`
   推送到 GitHub

---

## 4. 当前项目最常用命令

先进入项目目录：

```powershell
cd D:\project\project\mindpal_demo\agent_version
```

查看状态：

```powershell
git status
```

查看改动内容：

```powershell
git diff
```

提交本次修改：

```powershell
git add .
git commit -m "fix api contract"
git push
```

查看提交历史：

```powershell
git log --oneline
```

---

## 5. 怎么看 Git 状态

执行：

```powershell
git status
```

你会看到类似结果：

```powershell
modified:   backend/api_endpoints.py
modified:   frontend/frontend.py
```

这表示文件被修改了，但还没有提交。

常见状态：

- `modified`：文件被修改
- `new file`：新文件
- `deleted`：文件被删除
- `staged`：已经加入暂存区，准备提交

---

## 6. 怎么看自己到底改了什么

查看所有改动：

```powershell
git diff
```

只看某个文件：

```powershell
git diff backend\api_endpoints.py
```

建议养成习惯：

- 提交前先 `git diff`
- 确认没有把无关内容一起提交进去

---

## 7. 怎么提交代码

### 提交全部改动

```powershell
git add .
git commit -m "fix recommendation api"
```

### 只提交某个文件

```powershell
git add backend\api_endpoints.py
git commit -m "fix content recommend endpoint"
```

### 一次提交的建议

一次提交尽量只做一类事情，比如：

- 修一个 bug
- 增加一个接口
- 调整一份文档

不建议一次提交里混太多不相关内容。

---

## 8. commit 信息怎么写

推荐写法：

```powershell
git commit -m "fix chat response contract"
git commit -m "add startup scripts"
git commit -m "update interview presentation"
```

好的提交说明特点：

- 短
- 清楚
- 一眼知道干了什么

不推荐这种：

```powershell
git commit -m "update"
git commit -m "change"
git commit -m "123"
```

---

## 9. 推送到 GitHub

当前项目已经完成首次推送，后续一般直接：

```powershell
git push
```

如果是第一次推某个新分支，通常写：

```powershell
git push -u origin 分支名
```

当前主分支是：

```powershell
main
```

所以主分支第一次推送时是：

```powershell
git push -u origin main
```

---

## 10. 从 GitHub 拉最新代码

拉取远程最新代码：

```powershell
git pull
```

如果你和别人一起开发，或者你在别的电脑上也改过，这个命令会很常用。

推荐习惯：

- 开始工作前先 `git pull`
- 工作完成后再 `git push`

---

## 11. 查看提交历史

最简洁常用：

```powershell
git log --oneline
```

示例：

```powershell
2167bc6 init project
```

如果想图形化一点：

```powershell
git log --oneline --graph --decorate
```

---

## 12. 不小心 add 了怎么办

如果你执行了：

```powershell
git add .
```

但发现有些文件不该提交，可以取消暂存：

```powershell
git restore --staged 文件名
```

例如：

```powershell
git restore --staged backend\api_endpoints.py
```

如果想全部取消暂存：

```powershell
git restore --staged .
```

---

## 13. 文件改坏了，怎么撤回

如果文件还没有提交，只是工作区改乱了，可以恢复成最近一次提交的状态：

```powershell
git restore 文件名
```

例如：

```powershell
git restore frontend\frontend.py
```

注意：

- 这个命令会丢掉你还没提交的改动
- 用之前一定先确认

---

## 14. 只想看某个文件的历史

```powershell
git log -- backend\api_endpoints.py
```

如果还想看详细改动：

```powershell
git log -p -- backend\api_endpoints.py
```

---

## 15. 新手先会这 8 条就够了

```powershell
git status
git diff
git add .
git commit -m "message"
git push
git pull
git log --oneline
git restore --staged .
```

---

## 16. 分支是什么

分支就是一条独立开发线。

你可以理解成：

- `main`：主线，稳定版本
- `feature/xxx`：功能开发分支
- `fix/xxx`：bug 修复分支

新手前期只会用 `main` 也完全够用。

以后如果想学分支，最常见命令是：

查看分支：

```powershell
git branch
```

创建并切换到新分支：

```powershell
git checkout -b feature/my-change
```

切回主分支：

```powershell
git checkout main
```

---

## 17. 当前项目相关脚本

这个项目已经为你准备了固定解释器的启动脚本。

启动后端：

```powershell
.\run_backend.ps1
```

启动前端：

```powershell
.\run_frontend.ps1
```

这两个脚本默认使用：

```powershell
D:\anaconda3\envs\xwjpy_312\python.exe
```

---

## 18. 常见错误与处理

### 1. `nothing to commit, working tree clean`

说明当前没有新改动可以提交。

### 2. `Permission denied (publickey)`

说明 GitHub SSH 认证没配好，要检查 SSH key。

### 3. `Repository not found`

说明远程仓库不存在、地址写错，或者你没有权限。

### 4. `fatal: not a git repository`

说明你当前不在 Git 仓库目录里，要先 `cd` 到项目目录。

---

## 19. 给新手的实用建议

- 每完成一个小功能就提交一次
- 提交前先看 `git diff`
- 不懂时优先用 `git status`
- 先少量高频提交，不要攒一大坨
- 暂时不要乱用 `git reset --hard`
- `.env`、数据库、缓存文件不要提交到仓库

---

## 20. 你以后可以直接照抄的流程

### 场景一：改完代码准备提交

```powershell
cd D:\project\project\mindpal_demo\agent_version
git status
git diff
git add .
git commit -m "describe your change"
git push
```

### 场景二：今天开始工作前先同步

```powershell
cd D:\project\project\mindpal_demo\agent_version
git pull
```

### 场景三：不小心 add 多了

```powershell
git restore --staged .
```

### 场景四：某个文件改坏了，想恢复

```powershell
git restore 文件名
```

---

## 21. 一句话总结

如果只记一句话，就记这个：

```powershell
git status -> git diff -> git add . -> git commit -m "说明" -> git push
```

这就是你日常最核心的 Git 工作流。
