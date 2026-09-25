<script setup lang="ts">
import { computed, nextTick, ref } from 'vue'
import swanLake from '@/assets/hefei-swan-lake.jpg'
import {
  Building2,
  ChevronLeft,
  ChevronRight,
  Database,
  LogOut,
  MessageSquare,
  Search,
  Send,
  ShieldCheck,
  UserPlus,
} from 'lucide-vue-next'

type User = {
  id: string
  name: string
  org: string
  mobile: string
  role: string
}

type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
}

type ProjectResult = {
  project_id?: string
  project_name?: string
  name?: string
  title?: string
  region?: string
  city?: string
  industry?: string
  technology_direction?: string
  fused_score?: number
  metadata?: Record<string, unknown>
}

const savedUser = localStorage.getItem('anhui_user')
const user = ref<User | null>(savedUser ? JSON.parse(savedUser) : null)
const activeAuth = ref<'login' | 'register'>('login')
const sidebarCollapsed = ref(false)
const loading = ref(false)
const error = ref('')
const sessionId = ref(localStorage.getItem('anhui_session_id') || '')
const chatBoxRef = ref<HTMLElement | null>(null)

const loginForm = ref({
  mobile: '',
  password: '',
})

const registerForm = ref({
  name: '',
  organization: '',
  mobile: '',
  role: '企业项目负责人',
  password: '',
})

const demandForm = ref({
  region: '安徽',
  industry: '',
  stage: '',
  amount: '',
  need: '',
})

const query = ref('')
const messages = ref<ChatMessage[]>([
  {
    role: 'assistant',
    content: '您好，我是安徽科创项目智能推荐助手。请描述产业方向、技术需求、融资阶段或合作目标，我会从安徽科创项目库中检索并推荐项目。',
  },
])
const lastResults = ref<ProjectResult[]>([])
const lastTrace = ref<Record<string, unknown> | null>(null)

const isAuthed = computed(() => Boolean(user.value))

const pageStyle = computed(() => ({
  backgroundImage: `linear-gradient(180deg, rgba(245, 247, 250, 0.80), rgba(238, 243, 247, 0.92)), url(${swanLake})`,
}))

function persistUser(nextUser: User) {
  user.value = nextUser
  localStorage.setItem('anhui_user', JSON.stringify(nextUser))
}

function logout() {
  user.value = null
  localStorage.removeItem('anhui_user')
  localStorage.removeItem('anhui_session_id')
}

async function submitLogin() {
  error.value = ''
  loading.value = true
  try {
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(loginForm.value),
    })
    const data = await response.json()
    if (!response.ok) throw new Error(data.detail || '登录失败')
    persistUser(data.user)
  } catch (err) {
    error.value = err instanceof Error ? err.message : '登录失败'
  } finally {
    loading.value = false
  }
}

async function submitRegister() {
  error.value = ''
  loading.value = true
  try {
    const response = await fetch('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(registerForm.value),
    })
    const data = await response.json()
    if (!response.ok) throw new Error(data.detail || '注册失败')
    persistUser(data.user)
  } catch (err) {
    error.value = err instanceof Error ? err.message : '注册失败'
  } finally {
    loading.value = false
  }
}

function buildDemandQuery() {
  const parts = [
    demandForm.value.region && `区域：${demandForm.value.region}`,
    demandForm.value.industry && `技术方向：${demandForm.value.industry}`,
    demandForm.value.stage && `项目阶段：${demandForm.value.stage}`,
    demandForm.value.amount && `融资金额：${demandForm.value.amount}`,
    demandForm.value.need && `具体需求：${demandForm.value.need}`,
  ].filter(Boolean)
  query.value = `请从安徽科创项目库中推荐符合条件的项目。${parts.join('；')}。请给出推荐理由、项目名称和关键信息。`
}

async function sendMessage(text?: string) {
  const content = (text || query.value).trim()
  if (!content || loading.value || !user.value) return

  messages.value.push({ role: 'user', content })
  query.value = ''
  loading.value = true
  error.value = ''
  await scrollToBottom()

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: content,
        top_k: 5,
        session_id: sessionId.value || undefined,
        user_id: user.value.id || user.value.mobile,
      }),
    })
    const data = await response.json()
    if (!response.ok) throw new Error(data.detail || '推荐服务暂时不可用')
    sessionId.value = data.session_id
    localStorage.setItem('anhui_session_id', data.session_id)
    lastResults.value = Array.isArray(data.results) ? data.results : []
    lastTrace.value = {
      intent: data.intent,
      retrieval_mode: data.retrieval_mode,
      answer_mode: data.answer_mode,
      trace_id: data.trace_id,
    }
    messages.value.push({
      role: 'assistant',
      content: data.answer || '已完成检索，但没有生成有效回答。',
    })
  } catch (err) {
    const message = err instanceof Error ? err.message : '请求失败'
    error.value = message
    messages.value.push({ role: 'assistant', content: `抱歉，${message}。` })
  } finally {
    loading.value = false
    await scrollToBottom()
  }
}

async function scrollToBottom() {
  await nextTick()
  if (chatBoxRef.value) {
    chatBoxRef.value.scrollTop = chatBoxRef.value.scrollHeight
  }
}

function projectName(item: ProjectResult) {
  return item.project_name || item.name || item.title || item.metadata?.project_name || item.project_id || '未命名项目'
}

function scoreText(score?: number) {
  return typeof score === 'number' ? `${Math.round(score * 100)}%` : '已召回'
}
</script>

<template>
  <main class="app-shell" :style="pageStyle">
    <section v-if="!isAuthed" class="auth-page">
      <div class="auth-brand">
        <div class="brand-mark">
          <Building2 :size="30" />
        </div>
        <p>安徽科创项目库</p>
        <h1>安徽科创项目智能推荐助手</h1>
        <span>面向企业技术合作、项目筛选和投融资线索检索的智能服务入口</span>
      </div>

      <div class="auth-card">
        <div class="auth-tabs">
          <button :class="{ active: activeAuth === 'login' }" @click="activeAuth = 'login'">登录</button>
          <button :class="{ active: activeAuth === 'register' }" @click="activeAuth = 'register'">注册</button>
        </div>

        <form v-if="activeAuth === 'login'" class="auth-form" @submit.prevent="submitLogin">
          <label>手机号<input v-model="loginForm.mobile" placeholder="请输入 11 位手机号" /></label>
          <label>密码<input v-model="loginForm.password" type="password" placeholder="不少于 8 位" /></label>
          <p v-if="error" class="error-text">{{ error }}</p>
          <button class="primary-btn" :disabled="loading">
            <ShieldCheck :size="18" /> {{ loading ? '登录中...' : '进入工作台' }}
          </button>
        </form>

        <form v-else class="auth-form" @submit.prevent="submitRegister">
          <label>姓名<input v-model="registerForm.name" placeholder="请输入姓名" /></label>
          <label>单位<input v-model="registerForm.organization" placeholder="企业、高校或服务机构名称" /></label>
          <label>手机号<input v-model="registerForm.mobile" placeholder="请输入 11 位手机号" /></label>
          <label>身份角色<input v-model="registerForm.role" placeholder="例如：企业项目负责人" /></label>
          <label>密码<input v-model="registerForm.password" type="password" placeholder="不少于 8 位" /></label>
          <p v-if="error" class="error-text">{{ error }}</p>
          <button class="primary-btn" :disabled="loading">
            <UserPlus :size="18" /> {{ loading ? '注册中...' : '注册并进入' }}
          </button>
        </form>
      </div>
    </section>

    <section v-else class="workspace">
      <aside class="sidebar" :class="{ collapsed: sidebarCollapsed }">
        <div class="sidebar-head">
          <div class="brand-mark small"><Building2 :size="22" /></div>
          <div v-if="!sidebarCollapsed">
            <strong>安徽科创助手</strong>
            <span>项目推荐工作台</span>
          </div>
          <button class="icon-btn" @click="sidebarCollapsed = !sidebarCollapsed">
            <ChevronLeft v-if="!sidebarCollapsed" :size="18" />
            <ChevronRight v-else :size="18" />
          </button>
        </div>
        <nav>
          <button class="nav-item active"><MessageSquare :size="18" /><span v-if="!sidebarCollapsed">智能对话</span></button>
          <button class="nav-item"><Search :size="18" /><span v-if="!sidebarCollapsed">项目检索</span></button>
          <button class="nav-item"><Database :size="18" /><span v-if="!sidebarCollapsed">项目库状态</span></button>
        </nav>
        <div class="user-box" v-if="!sidebarCollapsed">
          <strong>{{ user?.name }}</strong>
          <span>{{ user?.org }}</span>
          <button @click="logout"><LogOut :size="16" /> 退出登录</button>
        </div>
      </aside>

      <div class="main-panel">
        <header class="topbar">
          <div>
            <p>安徽省科技创新项目智能服务</p>
            <h2>安徽科创项目智能推荐助手</h2>
          </div>
          <div class="status-pill">RAG 项目库已接入</div>
        </header>

        <section class="content-grid">
          <div class="intake-panel">
            <h3>需求表单</h3>
            <label>区域<input v-model="demandForm.region" /></label>
            <label>技术方向<input v-model="demandForm.industry" placeholder="如新能源、人工智能、低空经济" /></label>
            <label>项目阶段<input v-model="demandForm.stage" placeholder="如中试、产业化、融资中" /></label>
            <label>融资/合作要求<input v-model="demandForm.amount" placeholder="如 5000 万以上、股权融资" /></label>
            <label>补充需求<textarea v-model="demandForm.need" rows="4" placeholder="描述企业诉求、应用场景或筛选条件"></textarea></label>
            <button class="secondary-btn" @click="buildDemandQuery">生成推荐问题</button>
            <button class="primary-btn" @click="sendMessage()" :disabled="loading || !query.trim()">提交推荐</button>
          </div>

          <div class="chat-panel">
            <div ref="chatBoxRef" class="messages">
              <article v-for="(item, index) in messages" :key="index" class="message" :class="item.role">
                <div class="avatar">{{ item.role === 'user' ? '我' : '科' }}</div>
                <p>{{ item.content }}</p>
              </article>
              <article v-if="loading" class="message assistant">
                <div class="avatar">科</div>
                <p>正在识别需求、检索项目库并生成推荐...</p>
              </article>
            </div>
            <div class="quick-row">
              <button @click="sendMessage('推荐三个安徽新能源项目，并说明推荐理由')">新能源项目</button>
              <button @click="sendMessage('找一下低空经济相关项目')">低空经济</button>
              <button @click="sendMessage('介绍一下刚才推荐的第一个项目')">追问第一个</button>
            </div>
            <div class="composer">
              <textarea v-model="query" rows="2" placeholder="直接输入需求，例如：帮我找三个安徽人工智能项目，比较融资情况，再介绍第一个项目。" @keydown.ctrl.enter.prevent="sendMessage()" />
              <button class="send-btn" :disabled="loading || !query.trim()" @click="sendMessage()"><Send :size="18" /></button>
            </div>
          </div>

          <aside class="result-panel">
            <h3>本轮召回项目</h3>
            <div v-if="lastResults.length" class="result-list">
              <article v-for="(item, index) in lastResults" :key="item.project_id || index">
                <span>{{ index + 1 }}</span>
                <div>
                  <strong>{{ projectName(item) }}</strong>
                  <p>{{ item.industry || item.technology_direction || item.city || item.region || '安徽科创项目' }}</p>
                </div>
                <em>{{ scoreText(item.fused_score) }}</em>
              </article>
            </div>
            <p v-else class="empty-text">提交需求后，这里会显示 RAG 召回的项目。</p>
            <div v-if="lastTrace" class="trace-box">
              <strong>识别与路由</strong>
              <pre>{{ JSON.stringify(lastTrace, null, 2) }}</pre>
            </div>
          </aside>
        </section>
      </div>
    </section>
  </main>
</template>
