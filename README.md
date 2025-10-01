
# Python 推理服务部署到 Sealos 全流程

本文档介绍如何从零开始，将本地的 Python 推理服务（FastAPI + PyTorch）打包成 Docker 镜像，并部署到 Sealos 集群中，通过 Service 供前端调用。

---

## 📦 1. 本地准备代码与依赖

### 1. 确保你的 Python 项目结构如下：



   app/

   │── app.py                 # FastAPI 推理服务入口

   │── requirements.txt       # 依赖文件

   │── nets/                  # 模型代码

   │── util/                  # 工具代码

   │── checkpoint/            # 模型权重文件

   │── Dockerfile             # 镜像构建文件

---

## 🛠️ 2. 构建并测试 Docker 镜像

### 1. 在项目目录执行构建：

```bash
docker build -t py-infer:latest .
```

### 2. 启动容器并映射端口：

```bash
docker run --rm -p 8000:8000 py-infer:latest
```

### 3. 测试推理接口：

```bash
curl -X POST "http://127.0.0.1:8000/infer" \
     -F "file=@test.jpg"  #根据具体需要post的内容修改
```
返回结果符合预期，即说明服务正常。

或者：

- 打开浏览器：http://127.0.0.1:8000/docs → 发送 POST 请求 → 正常得到返回即成功
---

## ☁️ 3. 推送镜像到 Docker Hub

### 1. 登录 Docker Hub：

- 登录 Docker Hub 网站

- 右上角头像 → Account Settings → Security → New Access Token。

- 取个名字，点击 Generate。

- 复制生成的一长串 token。

- 回到 PowerShell，执行：

```bash
docker login
```
- Username: 你的 Docker Hub 用户名

- Password: 刚生成的 Access Token

### 2. 给镜像打标签：

```bash
docker tag py-infer:latest your_dockerhub_username/py-infer:latest
```

### 3. 推送镜像：

```bash
docker push your_dockerhub_username/py-infer:latest
```

### 4. 在 Docker Hub 仓库中可以看到镜像。

---

## 🚀 4. 部署到 Sealos 集群
### 1. 检查本机是否装了 kubectl

```bash
kubectl version --client
```

### 2. 从 Sealos 控制台下载kubeconfig.yaml
 
### 3. 设置环境变量，在 PowerShell 执行：
```bash
$env:KUBECONFIG = "C:\kube\kubeconfig.yaml"（改为你的路径）
```
### 4. 查看当前命名空间
```bash
kubectl config view --minify -o "jsonpath={..namespace}"
```
### 5. 查询是否能访问该命名空间
```bash
kubectl get pods -n user-system（user-system为之前上一步查询到的名字）
```
- 能显示结果（哪怕是空列表而不是 Forbidden），说明你在 user-system 有权限

### 6. 编写 Kubernetes 部署文件

- 在本地新建 `py-infer.yaml`：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: py-infer #你的镜像名
  namespace: ns-5vx9sy9v  #你的user-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: py-infer
  template:
    metadata:
      labels:
        app: py-infer
    spec:
      # ★★★ Pod 级安全上下文（满足 restricted 策略）
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        seccompProfile:
          type: RuntimeDefault

      containers:
        - name: py-infer
          image: docker.io/elysialover/py-infer:latest #你的docker hub用户名/镜像名:Tag
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000

          # ★★★ 容器级安全上下文（满足 restricted 策略）
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
            # readOnlyRootFilesystem 可选；部分库会写缓存，先不启用，若策略强制再说
            # readOnlyRootFilesystem: true

          readinessProbe:
            httpGet: { path: /healthz, port: 8000 }
            initialDelaySeconds: 20
            periodSeconds: 5
            failureThreshold: 6       # 可选：最多连续失败 6 次
            timeoutSeconds: 2  
          livenessProbe:
            httpGet: { path: /healthz, port: 8000 }
            initialDelaySeconds: 40
            periodSeconds: 10
            failureThreshold: 3
            timeoutSeconds: 2
          resources:
            requests: { cpu: "100m", memory: "256Mi" }
            limits:   { cpu: "1",    memory: "1Gi" }
---
apiVersion: v1
kind: Service
metadata:
  name: py-infer   #你的镜像名
  namespace: ns-5vx9sy9v  #你的user-system
spec:
  selector: { app: py-infer }
  ports:
    - name: http
      port: 8000
      targetPort: 8000
  type: ClusterIP
```

### 7. 应用配置

```bash
kubectl apply -f py-infer.yaml
```

检查状态：

```bash
kubectl -n ns-xxxxxxx get pods -o wide
kubectl -n ns-xxxxxxx get svc/py-infer 18000:8000
```
- Pod 状态变为 Running
- 打开 http://127.0.0.1:18000/healthz → 应返回 {"ok": true}
- 打开 http://127.0.0.1:18000/docs → 进行 POST 尝试
---

## 🔗 5. Python服务访问地址

- 若前后端容器与Python服务在同一命名空间：
```bash
http://py-infer:8000/infer
```
- 若不在同一命名空间
```bash
http://py-infer.ns-5vx9sy9v.svc.cluster.local:8000/infer
```

---

## ✅ 总结流程

1. 写好 `requirements.txt` 与 `Dockerfile`。
2. 本地构建并测试镜像。
3. 推送镜像到 Docker Hub。
4. 在 Sealos 写 `py-infer.yaml`，Deployment + Service。
5. 用 `kubectl apply` 部署。

这样就完成了 **从本地代码 → Docker Hub → Sealos 集群 → 前后端访问** 的全流程。


