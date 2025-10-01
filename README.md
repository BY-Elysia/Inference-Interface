
# Python 推理服务部署到 Sealos 全流程

本文档介绍如何从零开始，将本地的 Python 推理服务（FastAPI + PyTorch）打包成 Docker 镜像，并部署到 Sealos 集群中，通过 Service 供前端调用。

---

## 📦 1. 本地准备代码与依赖

1. 确保你的 Python 项目结构如下：



app/

│── app.py                 # FastAPI 推理服务入口

│── requirements.txt       # 依赖文件

│── nets/                  # 模型代码

│── util/                  # 工具代码

│── checkpoint/            # 模型权重文件

│── Dockerfile             # 镜像构建文件


## 🛠️ 2. 构建并测试 Docker 镜像

1. 在项目目录执行构建：

```bash
docker build -t py-infer:latest .
```

2. 启动容器并映射端口：

```bash
docker run --rm -p 8000:8000 py-infer:latest
```

3. 测试推理接口：

```bash
curl -X POST "http://127.0.0.1:8000/infer" \
     -F "file=@test.jpg"
```

返回结果包含类别和 `imageBase64` 图片，说明服务正常。

---

## ☁️ 3. 推送镜像到 Docker Hub

1. 登录 Docker Hub：

```bash
docker login
```

2. 给镜像打标签：

```bash
docker tag py-infer:latest your_dockerhub_username/py-infer:v1
```

3. 推送镜像：

```bash
docker push your_dockerhub_username/py-infer:v1
```

4. 在 Docker Hub 仓库中可以看到镜像。

---

## 🚀 4. 部署到 Sealos 集群

### 4.1 编写 Kubernetes 部署文件

在本地新建 `py-infer.yaml`：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: py-infer
  namespace: ns-xxxxxxx   # 你的 namespace
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
      containers:
        - name: py-infer
          image: your_dockerhub_username/py-infer:v1
          ports:
            - containerPort: 8000
          securityContext:       # 满足 PodSecurity 限制
            runAsNonRoot: true
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
            seccompProfile:
              type: RuntimeDefault

---
apiVersion: v1
kind: Service
metadata:
  name: py-infer
  namespace: ns-xxxxxxx   # 你的 namespace
spec:
  selector:
    app: py-infer
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP
```

### 4.2 应用配置

```bash
kubectl apply -f py-infer.yaml
```

检查状态：

```bash
kubectl -n ns-xxxxxxx get pods
kubectl -n ns-xxxxxxx get svc
```

---

## 🔗 5. 前端访问服务

1. 本地端口转发：

```bash
kubectl -n ns-xxxxxxx port-forward svc/py-infer 18000:8000
```

现在可通过 `http://127.0.0.1:18000/infer` 访问。

2. 如果需要外部访问，可将 Service 类型改为 `NodePort` 或 `LoadBalancer`。

---

## ✅ 总结流程

1. 写好 `requirements.txt` 与 `Dockerfile`。
2. 本地构建并测试镜像。
3. 推送镜像到 Docker Hub。
4. 在 Sealos 写 `py-infer.yaml`，Deployment + Service。
5. 用 `kubectl apply` 部署。
6. 用 `port-forward` 或 `NodePort` 暴露服务，前端调用。

这样就完成了 **从本地代码 → Docker Hub → Sealos 集群 → 前端访问** 的全流程。

```

---

要不要我帮你再生成一个**完整可运行的 `py-infer.yaml` 文件**（替换好 namespace 和镜像名），直接可以 `kubectl apply`？
```
