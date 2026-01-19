# GA4 Studio Dashboard (Streamlit)

Streamlit-приложение для выгрузки GA4:
- отчёт по списку URL/путей
- итоги по сайту за период

Секреты НЕ хранятся в репозитории: используется Streamlit Secrets.

---

## 1) Требования
- Python 3.10+ (локально)
- GA4 Property ID
- Google Service Account с доступом к GA4 (роль Viewer/Read & Analyze)

---

## 2) Секреты (обязательно)

Нужны 2 секрета:

### 2.1 GA4 Property ID
`GA4_PROPERTY_ID="123456789"`

### 2.2 Service Account (в виде словаря)
`gcp_service_account` — это поля из JSON сервис-аккаунта, но НЕ файл.

---

## 3) Локальный запуск

### 3.1 Установка
```bash
pip install -r requirements.txt
