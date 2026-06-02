import { createI18n } from 'vue-i18n'
import languages from '../../../locales/languages.json'

const localeFiles = import.meta.glob('../../../locales/!(languages).json', { eager: true })

const messages = {}
const availableLocales = []

for (const path in localeFiles) {
  const key = path.match(/\/([^/]+)\.json$/)[1]
  if (languages[key] && key !== 'zh') {
    messages[key] = localeFiles[path].default
    availableLocales.push({ key, label: languages[key].label })
  }
}

const savedLocaleRaw = localStorage.getItem('locale')
const savedLocale = (savedLocaleRaw && savedLocaleRaw !== 'zh' && messages[savedLocaleRaw])
  ? savedLocaleRaw
  : 'en'

if (savedLocaleRaw !== savedLocale) {
  localStorage.setItem('locale', savedLocale)
}

const i18n = createI18n({
  legacy: false,
  locale: savedLocale,
  fallbackLocale: 'en',
  messages
})

export { availableLocales }
export default i18n
