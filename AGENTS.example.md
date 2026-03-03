# AGENTS / agents.md — 本仓库 AI/Agent Go 编码与改动规范（简体中文）

> 你是任何形式的 AI/Agent（Copilot、Cursor、LLM Agent、自动重构工具等）。只要你在本仓库写/改 Go 代码，就必须遵守本文。
>
> 主要痛点（必须被解决）：**过度抽象、过度封装、乱定义接口、过度防御性编程（无意义的判空/判空 slice 等）、缺少注释、参数/返回值过多、频繁重构**。
>
> 目标：产出 **简单直接、可读可审、可维护可测试** 的 Go 代码，而不是“看起来很架构”的代码。

---

## 1. 仓库事实与目录结构（先对齐背景）

- Go Monorepo（根目录 `go.mod`），主要栈：**go-zero RPC / gRPC / protoc-gen-validate / GORM Gen / OpenTelemetry**。
- 微服务目录：`service/<svc>/`（例如 `service/user`、`service/authz`）。
- 典型结构（按职责从外到内）：
  - `service/<svc>/internal/server/...`：RPC 适配层（常由 goctl 生成）；只负责“转发 + 适配”，不要塞业务。
  - `service/<svc>/internal/logic/...`：业务编排层（校验、权限语义、调用 repo/外部 RPC）。
  - `service/<svc>/internal/repo/...`：数据/外部依赖访问层（DB、Authz RPC 等）。
  - `service/<svc>/internal/dal/...` + `.../dal/query/...`：GORM Gen 数据层；业务写入通常通过 `q.Transaction(...)`。
  - `pkg/...`：跨服务基础能力（**必须保持业务无关**）。
- 协议：`idl/` 放 proto；生成代码在 `service/<svc>/pb/...`（以及可能存在的公共 `pb/`、`client/` 目录）。
- 配置：`service/<svc>/etc/*.yaml`（常带 `.example`）；代码读取配置以 `svcCtx.Config` 为准。

---

## 2. 生成代码的铁律（不要对抗生成器）

1. **禁止手改生成文件**（除非用户明确要求且理解后果）：
  - `*.pb.go`、`*_grpc.pb.go`、`*.pb.validate.go`
  - 含 `// Code generated ... DO NOT EDIT.` 的文件（如 goctl 生成的 server）。
2. 改 RPC/协议：先改 `idl/*.proto`，再用脚本/工具重新生成。
3. 改动“生成文件”时，你必须在回复中说明：**为什么不能通过修改 proto/模板生成来解决**。

---

## 3. 开发/自检命令（你写的代码至少要经得起这些）

- `gofmt -w .`
- `go vet ./...`
- `golangci-lint run ./... --timeout=5m`（如仓库/CI 启用）
- `go test ./...`（关键路径/并发建议加 `-race`）

---

## 4. AI 常见坏味道（黑名单：硬性禁止）

### 4.1 过度封装/过度抽象（最常见）

**以下行为一律禁止：**

- 把只用一次的小逻辑拆成 helper，造成读者理解需要来回跳转（“跳转地狱”）。
- 为了“看起来整洁”新增一层包/一层结构体/一层接口/一层适配器。
- 新增 `utils/`、`helpers/`、`common/` 之类“垃圾抽屉包”用来堆零散函数（除非仓库已有且团队约定这么做）。
- 预判未来需求而提前设计扩展点（工厂/策略/插件/过早泛化）。
- “为了重构而重构”：在同一次改动里顺手改目录、批量改命名、批量抽象。
- 无必要使用泛型/反射/复杂设计模式（只为“高级”或“优雅”）。
- **过度防御性编程**：到处 `if x == nil` / `if len(s) > 0` / `if ctx == nil` 兜底；输入校验请按第 8.4 节“Validate Once”做在边界即可。
- **过度错误分类/解包**：为了“看起来专业”而到处写 `errors.Is/As`、类型断言、字符串匹配来识别错误来源（例如判断“某个字段缺失”）——除非本仓库已有稳定约定且确实需要分支处理（见第 8.5 节）。

### 4.2 频繁重构（破坏 review）

除非用户明确要求“重构/架构升级/规范化”，否则：

- 不移动目录
- 不批量改命名
- 不做大范围拆分/合并文件
- 不把“行为变更”和“大重构”混在同一次修改

---

## 5. Go 风格与命名（强制）

### 5.1 命名规则（强制）

- **局部变量/参数**：`lowerCamelCase`。禁止 `OLD/NEW/ADD/DEL/SlicePermissions` 这种大写局部变量（大写只用于导出标识符或常量）。
- **导出标识符**：`PascalCase`。
- **包名**：短、清晰、语义明确；避免 `common/util/base`。
- **错误日志上下文**：用“动词 + 对象”的短语，例如 `"db create role failed"`、`"authz UpdateRules failed"`。

### 5.2 结构与放置（就近原则）

- 业务编排：`internal/logic/...`
- DB/外部 RPC：`internal/repo/...`
- 跨服务基础能力：`pkg/...`（必须业务无关）

不要为了抽象把代码搬到“更远”的地方。

### 5.3 配置与依赖（强制）

- 读取配置：**只用 `svcCtx.Config`**，不要在业务逻辑里 `os.Getenv`、读取文件路径、读 `.env`。
- 新增依赖：默认禁止；确需引入第三方库时，必须说明“为什么标准库/现有依赖不够”。

---

## 6. 抽象/封装/接口引入门槛（必须过闸）

当你想新增 **helper / 抽象 / interface / 新包** 时，你必须在回复里写出“过闸理由”。

### 6.1 新增 helper 函数（必须满足至少 2 条）

1. 同样逻辑已在仓库中出现至少 **2 处重复**（或本次变更会落地第 2 处）。
2. helper 承载明确不变量/边界（校验、归一化、资源生命周期、状态转换）。
3. helper 显著提升可测试性（隔离时间/随机/外部依赖）。
4. helper 显著降低认知复杂度（复杂分支、复杂状态机、协议解析）。

不满足：直接内联。

### 6.2 新增 interface（必须满足至少 1 条，且要写清楚）

1. 现在就存在 **至少 2 个实现**（真实存在，不是臆想）。
2. 为测试隔离外部依赖：优先 `httptest`/内存实现；只有当它们不合适才引入 interface。
3. 需要稳定模块边界且团队认可（跨包契约、插件点等）。

> 说明：本仓库**允许**“单方法小接口”（例如 repo 依赖注入、适配外部 RPC/DB）。但它必须带来明确收益：
> - 由使用方（consumer）定义；
> - 方法集最小（通常 1～3 个，单方法也完全 OK）；
> - **不要**为了“看起来面向接口”而给每个 struct 都配一个 interface。

额外硬规则：
- interface **由使用方定义**（consumer side），方法集最小（通常 1～3 个）。
- 禁止“每个 struct 配一个 interface”。

---

## 7. 函数签名控制（解决参数/返回值过多）

### 7.1 参数数量（强制）

- 优先：0～3 个参数。
- 当参数 ≥ 4：默认改为 `Options/Request` 结构体。

✅ 推荐：
- `type CreateRoleOptions struct { ... }`
- `type UpdateRoleRequest struct { ... }`

❌ 禁止：
- 为了“高级”而使用 functional options（除非可选项很多且调用方很多）。

### 7.2 返回值数量（强制）

- 推荐：`(T, error)` / `error`。
- 当返回值 ≥ 3：必须改为 `Result struct + error`。

> 对接既有 API（例如历史遗留多返回值）可以保留，但**不得新增新的多返回值公共函数**。

---

## 8. 错误处理、日志与链路（结合本仓库约定）

### 8.1 错误返回（强制）

- 参数/校验错误：
  - `xerr.NewErrCodeMsg(xerr.REQUEST_PARAM_ERROR, err.Error())`
- 系统/依赖错误：
  - 对外 `errMsg` 必须友好、避免泄露内部细节；
  - 对内必须补齐：
    - `.WithLogMsg("...")`
    - `.WithCause(err)`
- **已是对外错误则原样返回**：如果拿到的 `err` 已经是本仓库约定的对外错误（例如 `xerr.CodeError`），直接 `return ..., err`；不要二次包装、不要改写文案、更不要再做 `errors.Is/As` 去重新分类。

✅ 推荐模板：
```go
if err := in.Validate(); err != nil {
return nil, xerr.NewErrCodeMsg(xerr.REQUEST_PARAM_ERROR, err.Error()).
WithLogMsg("invalid request param").
WithCause(err)
}

res, err := l.svcCtx.Store.SomeOp(ctx, l.svcCtx.Query, ...)
if err != nil {
return nil, xerr.NewErrCodeMsg(xerr.SERVER_COMMON_ERROR, "system error").
WithLogMsg("db SomeOp failed").
WithCause(err)
}
```

### 8.2 禁止重复日志（强制）

- 禁止在 logic/repo 里 `logx.Errorf(...)` 后又把错误返回。
  - 本仓库 RPC 拦截器会统一记录 `xerr.CodeError` 的 `logMsg/cause`。

### 8.3 Trace（可选但推荐）

- 新增关键 DB/外部调用路径时，优先用 `pkg/traceutil.Do(ctx, "span", func(ctx context.Context) error { ... })` 包裹。
- 但不要为了 trace 再抽一堆层。

### 8.4 校验边界与“少而准确”的防御性编程（强制）

你写的 Go 代码应该更像“人类写的”：**校验做在该做的地方，做到刚好够用**。不要把每个参数都用一串 `if` 包起来。

**核心原则：校验只做一次（Validate Once）**

- 只在“信任边界”（trust boundary）做输入校验：通常是 `internal/logic/...` 的 RPC 入口处（以及对外暴露的 `pkg` API 入口）。
- 一旦 `in.Validate()`（`protoc-gen-validate` 生成）通过，后续代码把字段当作**已满足前置条件**；不要在 repo/helper/内部循环里重复写 `Name == ""` / `Id == 0` / `len(slice) == 0` 之类守卫。
- 需要额外业务校验（跨字段/查重/权限语义等）时，把它们集中在入口处的一个“紧凑校验块”里做完，而不是散落在每一层。

#### 8.4.1 不要“重复判空 / 重复归一化”（强制）

本仓库最常见的 AI 坏味道之一：同一个参数在 **多层** 做同一件事（例如：入口 validate 过了，repo/helper 里又 `len==0`；caller 里做过去重/排序，helper 里又做一遍；或者 helper 里 `if len==0 return`，caller 里又 `if len(normalized)==0 return`）。

**规则（必须遵守）：**

- 对“同一个不变量”（非空、已去重、已排序、格式已规范化等），**只能选择一个地方负责**，其它层只依赖这个契约：
  - 要么放在 trust boundary（logic 入口）并通过 `Validate()`/业务校验保证；
  - 要么放在内部 helper（repo/包内私有函数）并在注释里声明它的输入契约。
- **禁止**在同一条调用链里“重复做两次”：
  - ❌ `validate(min_items=1)` 后，repo/helper 再写 `if len(ids)==0 { return nil }`（除非空集合在该 API 语义里是合法输入，见下一条）。
  - ❌ caller 先 `dedup/sort`，helper 再 `dedup/sort`（或者反过来）。
- 仅当“空集合是合法语义”时才允许 `len==0` 早退，并且需要在代码注释里写清楚（例如：`RoleIDs` 允许为空表示“清空绑定”，此时锁函数应当 no-op）。
- 如果你 **不确定** 某个 slice 是否可能为空，必须先在仓库里核对：
  - proto 的 validate 规则（是否有 `min_items: 1`）；
  - 业务注释/语义（空集合是否代表清空/跳过）；
  - 真实调用点是否可能传空。
- 写测试时同理：
  - 如果输入为空在语义上不可能发生，就不要为了“看起来更健壮”而加空输入用例；
  - 如果空输入是合法语义，则必须写用例覆盖这个分支。

**推荐写法（只做一次）**

```go
// 方案 A：入口保证非空，helper 假设契约成立（不重复判空）
// 说明：roleIDs 来自 req.Validate(min_items=1)，为空表示调用方编程错误。
func lockRolesForShare(ctx context.Context, q *query.Query, roleIDs []uint64) error {
    normalized := normalizeRoleIDs(roleIDs) // normalizeRoleIDs 只做“排序+去重”，不负责 len==0 守卫
    // ... 直接使用 normalized
}

// 方案 B：空集合是合法语义，则 helper 负责一次性早退（caller 不再重复判断）
func lockRolesForShare(ctx context.Context, q *query.Query, roleIDs []uint64) error {
    normalized := normalizeRoleIDs(roleIDs)
    if len(normalized) == 0 { // 只有当空集合是业务合法输入时才保留
        return nil
    }
    // ...
}
```

**哪些 `if` 属于“过度防御”（默认禁止）**

- `if ctx == nil { ctx = context.Background() }`（调用方传 `nil` 是编程错误，不要悄悄兜底）。
- `if req == nil { ... }`（除非你能证明某条调用链真的会传 `nil`；gRPC/RPC 入口通常不需要）。
- `if ids != nil && len(ids) > 0 { for ... }`（`range` 遍历 `nil` slice 安全，直接 `for range` 即可）。
- `if m != nil { _ = m[k] }`（读 `nil` map 安全；**写** `nil` map 才需要初始化）。
- 对已 `Validate()` 的字段重复判断空值/范围（例如再写一堆 `if in.Name == ""`）。

**哪些校验必须保留（该 `if` 还是得写）**

- 可能导致 panic/越界的地方：slice 下标访问、map 写入、对 optional 指针字段解引用、类型断言。
- 输入会影响数据一致性/安全的业务规则：例如权限边界、跨字段约束（A 与 B 二选一/必须同时出现）、幂等 key 格式等。
- repo 层仍要处理“系统错误”：DB/RPC/事务返回的 `error`（这是依赖错误，不是输入校验）。

**推荐写法（示例）**

```go
if err := in.Validate(); err != nil {
    return nil, xerr.NewErrCodeMsg(xerr.REQUEST_PARAM_ERROR, err.Error()).
        WithLogMsg("invalid request param").
        WithCause(err)
}

name := strings.TrimSpace(in.Name)
if name == "" {
    return nil, xerr.NewErrCodeMsg(xerr.REQUEST_PARAM_ERROR, "name is required").
        WithLogMsg("empty name after trim")
}

// 复杂规则集中写，必要时写一个“文件内私有函数” validateXxx(...) error
```

> 注：`validateXxx` 这种“单次使用校验函数”属于“承载明确不变量/边界”的 helper，符合 6.1 的过闸条件，不算过度抽象。

---

### 8.5 错误识别与分类（`errors.Is` / `errors.As`）（强制）

本仓库对错误处理的目标是：**语义清晰、最小分支、最小改动**。AI 常见的“过度工程”是为了给用户更“友好”的提示而去**解包/分类**底层错误（尤其是通过 `errors.As` 抓字段名、抓接口方法、或通过字符串匹配判断错误来源）。这类做法会让 review 变得困难、隐藏真实错误、并且很容易在升级依赖后失效。

#### 8.5.1 默认规则：不要识别，直接返回（或按模板统一包装）

- **默认禁止**新增任何 `errors.Is/As`/类型断言/字符串匹配，用来“猜测错误类型/来源”。
- 对于 **请求参数错误**：
  - 入口处 `in.Validate()` 失败时，直接按 8.1 模板返回：`xerr.NewErrCodeMsg(xerr.REQUEST_PARAM_ERROR, err.Error())...`。
  - **禁止**为了把某个字段（如 `Version`）改成自定义文案而额外写 `errors.As(...).Field() == "Version"` 之类的分支。
- 对于 **依赖/系统错误**：按 8.1 模板统一包装（`SERVER_COMMON_ERROR + system error + WithCause`），不要再细分“看起来更精确”的提示，除非用户明确要求。

> 一句话：**没有明确需求就别做错误分类**；保持 `if err != nil { return ..., errWrap }` 的直线型流程。

#### 8.5.2 允许使用 `errors.Is/As` 的少数场景（必须满足其一）

仅当你需要“真的改变控制流”时，才允许用 `errors.Is/As`：

1. **上下文取消/超时**：例如 `errors.Is(err, context.Canceled)` / `errors.Is(err, context.DeadlineExceeded)`，需要提前返回或转换为特定错误码。
2. **明确的哨兵错误（sentinel error）** 且仓库已有稳定约定：例如 DB 的 `ErrRecordNotFound`、RPC 的特定错误码等（以仓库现有代码为准）。
3. **幂等/并发/重试语义** 需要区分错误类型：例如“版本冲突/乐观锁冲突”确实需要返回冲突码、或需要决定是否重试。

#### 8.5.3 使用时的硬性约束（必须同时满足）

当你确实需要写 `errors.Is/As`：

- 必须在分支上方写一行中文注释，说明“为什么必须识别这个错误”（控制流/错误码/幂等语义）。
- **禁止**使用“临时接口”来做解包（例如 `var x interface{ Field() string }`），也禁止通过 `strings.Contains(err.Error(), "...")` 来判断错误类型。
- 必须补一个最小单测覆盖该分支（或在已有测试里加用例）。

> 说明：如果你拿不到明确的错误类型（只能靠字符串/临时接口猜），那就说明该分支不够稳定 —— 按 8.5.1 的默认规则直接返回/统一包装。

---

## 9. 数据库与事务（GORM Gen 约定）

- DB 访问必须走 `svcCtx.Query`（GORM Gen），不要在业务里到处拿 `gorm.DB`。
- 多步写入必须在事务里：
  - `svcCtx.Query.Transaction(func(tx *query.Query) error { ... })` 或现有的 `svcCtx.Store.InTX(...)`。
- repo 层只做“访问与持久化”，不要把业务规则塞进 repo。

---

## 10. 权限/鉴权约定（Authz + perm）

- `AuthzInterceptor` 会把 gRPC 方法名转换成 `strings.ToLower(MethodName)` 作为 resource。
- 新增 RPC 方法且需要权限控制时：
  1. 在对应 logic 文件 `init()` 注册资源：`perm.AddResource(ResourceDef{ Key: strings.ToLower(methodName), ... })`。
  2. 通常 `AllowOperations` 包含 `call`。
- 若新增方法应该免鉴权：更新白名单（`pkg/interceptor/rpcserver/authzInterceptor.go` 的 `WhiteList`）。

---

## 11. 注释规范（解决“不写注释”）

### 11.1 必须写注释的地方（硬性）

1. **所有导出标识符**必须有 GoDoc（注释以名字开头）。
2. 复杂逻辑/非直觉分支必须解释“为什么”，不要翻译代码。
3. 安全/权限/鉴权相关逻辑必须说明意图与边界。

### 11.2 允许不写注释的地方

- 直白代码（注释只会重复）。


### 11.3 注释语言（强制）

- 本仓库所有注释（GoDoc、行内注释、块注释、TODO/FIXME、测试用例说明）默认一律使用**简体中文**。
- 不要为了“统一语言”去批量翻译不相关的历史英文注释；只要求你**本次新增/修改**的注释遵守本节规则（最小 diff）。
- 允许出现英文的场景仅限于：
  1. 代码标识符/包名/类型名/函数名（它们本来就是英文或缩写）；
  2. 必须保留的原文片段：SQL 关键字、协议字段名、第三方接口/错误信息原文、RFC/官方文档引用等；
  3. 行业内约定俗成且不翻译更清晰的术语（例如 `advisory lock`、`FOR UPDATE`），但必须在同一行或紧邻行给出中文解释。
- 禁止只写英文注释来“解释业务/意图”。如果需要提到英文术语，请写成：`英文术语（中文解释）`。
- 导出标识符的 GoDoc 仍必须“以名字开头”，但描述内容必须用中文：
  - ✅ `// LockUser 使用 PostgreSQL 事务级 advisory lock（事务级建议锁）串行化同一用户的写操作。`
  - ❌ `// LockUser serializes user writes using advisory lock.`

---

## 12. 测试（强烈推荐，关键改动必须写）

- 纯逻辑：table-driven 单测。
- 覆盖边界：空值、极值、错误分支。
- 不要为了测试到处造接口；优先用内存实现/httptest。

---

## 13. AI 输出工作流（你每次回复都必须按此顺序）

当用户要求你写/改 Go 代码时，你必须输出：

1. **方案摘要（5～10 行）**：只讲关键决策与边界，不讲架构大道理。
2. **最小可行改动（Minimal Diff）**：先让功能正确、能编译、能测试。
3. 如你引入 helper/interface/新包：必须写清楚“过闸理由”（见第 6 节）。
4. 提供最小用例：要么调用示例，要么 `*_test.go`。
5. 末尾附上自检清单（第 14 节），逐条勾选。

---

## 14. 最终自检清单（必须逐条勾选）

- [ ] 是否新增了只用一次的 helper？若有，已内联或给出过闸理由
- [ ] 若新增了 interface（含单方法小接口），是否满足第 6.2 节且已说明“带来的收益/必要性”
- [ ] 函数参数是否 ≥ 4？若是，是否改为 Options/Request 或解释原因
- [ ] 返回值是否 ≥ 3？若是，是否改为 Result struct 或解释原因
- [ ] 是否对内部错误补齐 `WithLogMsg` 与 `WithCause`
- [ ] 是否避免“log 后再 return error”的重复日志
- [ ] 是否避免“无意义的防御性 if”（对已 Validate 的字段重复判空、`ctx == nil` 兜底、`ids != nil && len(ids) > 0` 等）
- [ ] 是否避免“重复判空 / 重复归一化”（同一不变量只在一处负责：入口或 helper，不要两边都写）
- [ ] 是否避免“无意义的错误解包/分类”（不要为了改提示而新增 `errors.Is/As`、类型断言、字符串匹配；仅在确需改变控制流时使用，并满足第 8.5 节约束）
- [ ] 是否为导出标识符补齐 GoDoc 注释
- [ ] 注释是否全部为简体中文（除允许的例外外）
- [ ] 是否避免无意义重构（最小 diff）
- [ ] 是否包含必要测试（关键路径 + 错误路径）
- [ ] 是否 gofmt