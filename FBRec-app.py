import json
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. 資料庫架構 (Database Schema)
# ==========================================
Base = declarative_base()

class Tipster(Base):
    __tablename__ = 'tipsters'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    is_active = Column(Boolean, default=True) # 控制是否顯示 Tab
    created_at = Column(DateTime, default=datetime.now)
    tips = relationship("Tip", back_populates="tipster")

class Tip(Base):
    __tablename__ = 'tips'
    id = Column(Integer, primary_key=True)
    tipster_id = Column(Integer, ForeignKey('tipsters.id'), nullable=True)
    owner_type = Column(String(20), default='Tipster') # 分類: 'Tipster', 'User', 'System'
    
    category = Column(String(50))      
    tournament = Column(String(100))   
    home_team = Column(String(100))
    away_team = Column(String(100))
    market_type = Column(String(50))   
    line = Column(Float, default=0.0)  
    selection = Column(String(50))     
    odds = Column(Float, nullable=False)
    stake = Column(Float, default=0.0)    
    
    # 結算狀態
    home_score = Column(Integer, nullable=True)         
    away_score = Column(Integer, nullable=True)         
    home_ht_score = Column(Integer, nullable=True)      
    away_ht_score = Column(Integer, nullable=True)      
    home_corners = Column(Integer, nullable=True)       
    away_corners = Column(Integer, nullable=True)       
    home_ht_corners = Column(Integer, nullable=True)    
    away_ht_corners = Column(Integer, nullable=True)    
    
    status = Column(String(20), default='Open') 
    unit_profit = Column(Float, default=0.0) 
    profit = Column(Float, default=0.0)      
    payout = Column(Float, default=0.0)      
    
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    tipster = relationship("Tipster", back_populates="tips")

class BankrollLedger(Base):
    __tablename__ = 'bankroll_ledger'
    id = Column(Integer, primary_key=True)
    account_type = Column(String(20)) # 'Tipster', 'User', 'System'
    tipster_id = Column(Integer, ForeignKey('tipsters.id'), nullable=True)
    amount = Column(Float, nullable=False) 
    description = Column(String(200))
    created_at = Column(DateTime, default=datetime.now)

class ActionLog(Base):
    __tablename__ = 'action_logs'
    id = Column(Integer, primary_key=True)
    action_type = Column(String(50)) 
    record_id = Column(Integer)
    table_name = Column(String(50))
    old_data = Column(Text) 
    created_at = Column(DateTime, default=datetime.now)

# 用來隱藏不需要的下拉選單選項 (垃圾資料清理)
class HiddenOption(Base):
    __tablename__ = 'hidden_options'
    id = Column(Integer, primary_key=True)
    option_type = Column(String(20)) # 'team' or 'tournament'
    option_name = Column(String(100))

# 初始化資料庫
engine = create_engine('sqlite:///betting_system.db', echo=False, connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)

# 自動升級舊有資料庫 (Auto-Migration)
def upgrade_database(engine):
    inspector = inspect(engine)
    with engine.begin() as conn:
        try:
            if 'tips' in inspector.get_table_names():
                columns = [col['name'] for col in inspector.get_columns('tips')]
                if 'stake' not in columns: conn.execute(text("ALTER TABLE tips ADD COLUMN stake FLOAT DEFAULT 0.0"))
                if 'owner_type' not in columns: conn.execute(text("ALTER TABLE tips ADD COLUMN owner_type VARCHAR(20) DEFAULT 'Tipster'"))
            if 'tipsters' in inspector.get_table_names():
                t_columns = [col['name'] for col in inspector.get_columns('tipsters')]
                if 'is_active' not in t_columns: conn.execute(text("ALTER TABLE tipsters ADD COLUMN is_active BOOLEAN DEFAULT 1"))
        except Exception as e:
            pass # 防止重複執行報錯

upgrade_database(engine)
SessionLocal = sessionmaker(bind=engine)

# ==========================================
# 2. 核心算法與結算引擎
# ==========================================
def calculate_linked_odds(input_odds, margin=1.085):
    if input_odds <= 1.0: return 0.0
    prob_1 = 1 / input_odds
    prob_2 = margin - prob_1
    if prob_2 <= 0: return 0.0
    return round(1 / prob_2, 3)

def calculate_actual_margin(odds_1, odds_2):
    if odds_1 <= 1 or odds_2 <= 1: return 0.0
    return (1/odds_1) + (1/odds_2)

def get_opposite_selection(selection):
    mapping = {'主': '客', '客': '主', '大': '小', '小': '大'}
    return mapping.get(selection, selection)

def settle_asian_handicap(market_type, line, selection, 
                          h_sc, a_sc, h_ht_sc, a_ht_sc, 
                          h_cor, a_cor, h_ht_cor, a_ht_cor, odds=1.85):
    if market_type == "讓球":
        h_val, a_val = h_sc, a_sc
    elif market_type == "半場讓球":
        h_val, a_val = h_ht_sc, a_ht_sc
    elif market_type == "讓角":
        h_val, a_val = h_cor, a_cor
    elif market_type == "半場讓角":
        h_val, a_val = h_ht_cor, a_ht_cor
    elif market_type == "入球大小":
        total = h_sc + a_sc
    elif market_type == "半場入球大小":
        total = h_ht_sc + a_ht_sc
    elif market_type == "角球大小":
        total = h_cor + a_cor
    elif market_type == "半場角球大小":
        total = h_ht_cor + a_ht_cor

    if '大小' in market_type:
        diff = total - line
        is_over = (selection == '大')
        net_diff = diff if is_over else -diff
    else:
        diff = (h_val - a_val) + line
        is_home = (selection == '主')
        net_diff = diff if is_home else -diff

    if net_diff > 0.25: return 'Win', round(odds - 1.0, 3)
    elif net_diff == 0.25: return 'Half Win', round((odds - 1.0) / 2, 3)
    elif net_diff == 0.0: return 'Push', 0.0
    elif net_diff == -0.25: return 'Half Loss', -0.5
    else: return 'Loss', -1.0

# ==========================================
# 3. 機器學習與系統智能策略
# ==========================================
def get_historical_features(owner_type, tipster_id, session):
    query = session.query(Tip).filter(
        Tip.status != 'Open',
        Tip.is_deleted == False,
        Tip.owner_type == owner_type
    )
    if tipster_id: query = query.filter(Tip.tipster_id == tipster_id)
    
    tips = query.all()
    if len(tips) < 5: return None, None
    data = [{'odds': t.odds, 'line': t.line, 'target': 1 if t.status in ['Win', 'Half Win'] else 0} for t in tips]
    df = pd.DataFrame(data)
    return df[['odds', 'line']], df['target']

def analyze_system_edge(tipster_id, odds_tipster, odds_opp, current_line, sys_bankroll, session, kelly_fraction=0.25):
    X, y = get_historical_features('Tipster', tipster_id, session)
    
    if X is None or len(X) < 5:
        p_win_tipster = 0.5
        ai_status = "數據不足，採用基礎 50% 賠率機率評估"
    else:
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X, y)
        p_win_tipster = model.predict_proba([[odds_tipster, current_line]])[0][1]
        ai_status = "RF 模型運算完成，已挖掘歷史大數據！"

    # 系統自我學習機制 (如果有歷史系統投注數據，會微調機率)
    X_sys, y_sys = get_historical_features('System', None, session)
    if X_sys is not None and len(X_sys) >= 5:
        sys_win_rate = y_sys.mean()
        # 若系統勝率高，則增加對系統分析的信心加權
        ai_status += f" (結合系統自我學習，歷史系統勝率: {sys_win_rate*100:.1f}%)"

    # 計算對立面機率 (排除和局的簡化模型)
    p_win_opp = 1.0 - p_win_tipster
    
    # 計算期望值 EV
    ev_tipster = (p_win_tipster * odds_tipster) - 1.0
    ev_opp = (p_win_opp * odds_opp) - 1.0
    
    # 決定系統方向 (利益最大化)
    if ev_opp > ev_tipster and ev_opp > 0.02: # 加入微小門檻防呆
        sys_choice = 'Opposite'
        target_p = p_win_opp
        target_odds = odds_opp
    else:
        sys_choice = 'Tipster'
        target_p = p_win_tipster
        target_odds = odds_tipster

    # 凱利公式計算下注額
    b = target_odds - 1.0
    if b <= 0: return 0.0, p_win_tipster, p_win_opp, ev_tipster, ev_opp, sys_choice, "賠率異常"
    
    q = 1.0 - target_p
    kelly_f = max(0, (b * target_p - q) / b) * kelly_fraction
    suggested_stake = max(0, sys_bankroll * kelly_f)
    
    # 資金風控與上下限邏輯
    max_stake = sys_bankroll * 0.10 
    if suggested_stake > max_stake: suggested_stake = max_stake
    if suggested_stake < 10 and suggested_stake > 0: suggested_stake = 10

    return round(suggested_stake, 2), round(p_win_tipster, 4), round(p_win_opp, 4), round(ev_tipster, 4), round(ev_opp, 4), sys_choice, ai_status

# ==========================================
# 4. 財務統整與 UI 視覺函數
# ==========================================
def style_financials(val):
    if isinstance(val, (int, float)):
        color = 'red' if val < 0 else 'green' if val > 0 else 'gray'
        return f'color: {color}; font-weight: bold;'
    return ''

def get_avail_bankroll(account_type, tipster_id, session):
    ledgers_q = session.query(BankrollLedger).filter_by(account_type=account_type)
    if tipster_id: ledgers_q = ledgers_q.filter_by(tipster_id=tipster_id)
    ledgers = ledgers_q.all()
    
    tips_q = session.query(Tip).filter(Tip.status != 'Open', Tip.is_deleted == False, Tip.owner_type == account_type)
    if tipster_id: tips_q = tips_q.filter_by(tipster_id=tipster_id)
    tips = tips_q.all()
    
    dep = sum(l.amount for l in ledgers if l.amount < 0)
    wit = sum(l.amount for l in ledgers if l.amount > 0)
    profit = sum(t.profit for t in tips) 
    
    return abs(dep) - wit + profit

# ==========================================
# 5. 彈出視窗與資料庫管理 (Dialogs)
# ==========================================
@st.dialog("📊 數據庫即時預覽與管理", width="large")
def preview_db_dialog():
    session = SessionLocal()
    tipsters = session.query(Tipster).filter_by(is_active=True).all()
    t_names = [t.name for t in tipsters]
    
    main_tabs = st.tabs(["📝 各方投注記錄", "💰 各方資金流水", "⏪ 撤銷與回滾"])
    
    entities = [('System', '👑 系統', None), ('User', '👤 用家', None)] + [('Tipster', f'👥 {t.name}', t.id) for t in tipsters]
    
    # --- 投注記錄分類 ---
    with main_tabs[0]:
        sub_tabs = st.tabs([e[1] for e in entities])
        
        for i, (owner_type, label, t_id) in enumerate(entities):
            with sub_tabs[i]:
                query = session.query(Tip).filter(Tip.owner_type == owner_type, Tip.is_deleted == False)
                if t_id: query = query.filter(Tip.tipster_id == t_id)
                tips = query.order_by(Tip.id.desc()).all()
                
                if not tips:
                    st.info(f"{label} 尚無投注紀錄")
                    continue

                data = []
                for t in tips:
                    res = f"半場 {t.home_ht_score}-{t.away_ht_score} | 全場 {t.home_score}-{t.away_score}" if t.status != 'Open' else "未結算"
                    t_name_display = session.query(Tipster).get(t.tipster_id).name if t.tipster_id else "N/A"
                    data.append({
                        'ID': t.id, '分享者來源': t_name_display, '賽事': t.tournament, '對陣': f"{t.home_team} vs {t.away_team}",
                        '盤口': f"{t.market_type} ({t.line})", '選項': t.selection, '賠率': t.odds,
                        '投注額': t.stake, '結果': res, '狀態': t.status, '盈虧': t.profit
                    })
                
                df = pd.DataFrame(data)
                float_cols = ['賠率', '投注額', '盈虧']
                for c in float_cols: df[c] = df[c].astype(float)
                st.dataframe(df.style.map(style_financials, subset=['盈虧']).format({c: "{:.2f}" for c in float_cols}), use_container_width=True)
                
                c_del1, c_del2 = st.columns(2)
                with c_del1:
                    del_ids = st.multiselect("刪除特定注單 ID", df['ID'].tolist(), key=f"ms_tips_{owner_type}_{t_id}")
                    if st.button("執行刪除", key=f"bd_tips_{owner_type}_{t_id}"):
                        if del_ids:
                            tips_to_del = session.query(Tip).filter(Tip.id.in_(del_ids)).all()
                            old_data_list = [{'id': t.id, 'status': t.status, 'is_deleted': t.is_deleted} for t in tips_to_del]
                            for t in tips_to_del: t.is_deleted = True
                            session.add(ActionLog(action_type='BATCH_DELETE', record_id=0, table_name='tips', old_data=json.dumps(old_data_list)))
                            session.commit()
                            st.rerun()
                with c_del2:
                    confirm_clear = st.checkbox("確認一鍵清空", key=f"cc_tips_{owner_type}_{t_id}")
                    if st.button("清空紀錄", disabled=not confirm_clear, key=f"ca_tips_{owner_type}_{t_id}"):
                        all_t = query.all()
                        if all_t:
                            old_data_list = [{'id': t.id, 'status': t.status, 'is_deleted': t.is_deleted} for t in all_t]
                            for t in all_t: t.is_deleted = True
                            session.add(ActionLog(action_type='BATCH_DELETE', record_id=0, table_name='tips', old_data=json.dumps(old_data_list)))
                            session.commit()
                            st.rerun()

    # --- 資金流水分類 ---
    with main_tabs[1]:
        sub_tabs2 = st.tabs([e[1] for e in entities])
        for i, (owner_type, label, t_id) in enumerate(entities):
            with sub_tabs2[i]:
                query = session.query(BankrollLedger).filter_by(account_type=owner_type)
                if t_id: query = query.filter_by(tipster_id=t_id)
                ledgers = query.order_by(BankrollLedger.id.desc()).all()
                
                if ledgers:
                    ldf = pd.DataFrame([{
                        '時間': l.created_at.strftime('%Y-%m-%d %H:%M'),
                        '類型': '存入(負)' if l.amount < 0 else '提取(正)',
                        '金額': l.amount, '備註': l.description
                    } for l in ledgers])
                    ldf['金額'] = ldf['金額'].astype(float)
                    st.dataframe(ldf.style.map(style_financials, subset=['金額']).format({'金額': "{:.2f}"}), use_container_width=True)
                else:
                    st.info(f"{label} 尚無資金流水")
    
    # --- 撤銷中心 ---
    with main_tabs[2]:
        logs = session.query(ActionLog).order_by(ActionLog.id.desc()).limit(10).all()
        if logs:
            for log in logs:
                col1, col2 = st.columns([4, 1])
                if log.action_type == 'BATCH_DELETE':
                    records = json.loads(log.old_data)
                    col1.write(f"時間: {log.created_at.strftime('%m-%d %H:%M')} | 批量刪除 | 影響筆數: {len(records)}")
                else:
                    col1.write(f"時間: {log.created_at.strftime('%m-%d %H:%M')} | 動作: {log.action_type} | 紀錄 ID: {log.record_id}")
                    
                if col2.button("復原", key=f"undo_{log.id}"):
                    if log.action_type == 'BATCH_DELETE':
                        records = json.loads(log.old_data)
                        for item in records:
                            t = session.query(Tip).get(item['id'])
                            if t:
                                t.is_deleted = item.get('is_deleted', False)
                                t.status = item.get('status', 'Open')
                    else:
                        target_tip = session.query(Tip).get(log.record_id)
                        if target_tip:
                            old_data = json.loads(log.old_data)
                            for k, v in old_data.items():
                                setattr(target_tip, k, v)
                                
                    session.delete(log)
                    session.commit()
                    st.rerun()
        else:
            st.write("近期無可復原的操作。")
            
    session.close()

# ==========================================
# 6. 主程式與 UI 渲染 (Main App)
# ==========================================
def main():
    # 更改 Title 讓你確保看見版本已更新
    st.set_page_config(page_title="Tipster Quant AI v2.0", layout="wide")

    if 'odds1' not in st.session_state: st.session_state.odds1 = 1.85
    if 'odds2' not in st.session_state: st.session_state.odds2 = calculate_linked_odds(1.85)

    def update_odds1():
        st.session_state.odds2 = calculate_linked_odds(st.session_state.odds1)

    db_session = SessionLocal()
    tipsters = db_session.query(Tipster).filter_by(is_active=True).all()
    tipster_names = [t.name for t in tipsters]
    tipster_dict = {t.name: t.id for t in tipsters}

    tabs = st.tabs(tipster_names + ["➕ 新增分享者", "💰 資金總覽與結算", "⚙️ 系統管理"])

    # ----------------------------------------
    # [Tab] 新增分享者
    # ----------------------------------------
    with tabs[-3]:
        st.subheader("新增分享者")
        new_name = st.text_input("分享者稱號")
        if st.button("創建分享者"):
            exists = db_session.query(Tipster).filter_by(name=new_name).first()
            if exists:
                if not exists.is_active:
                    exists.is_active = True
                    db_session.commit()
                    st.success("分享者已從隱藏名單恢復！")
                    st.rerun()
                else:
                    st.error("該分享者已存在")
            elif new_name:
                db_session.add(Tipster(name=new_name))
                db_session.commit()
                st.success("創建成功！")
                st.rerun()

    # ----------------------------------------
    # [Tab] 系統管理
    # ----------------------------------------
    with tabs[-1]:
        st.subheader("系統管理員工具")
        
        st.button("🗄️ 開啟數據庫管理 (各方注單/流水/撤銷)", on_click=preview_db_dialog, type="primary")
        
        if st.button("🔄 清除前端快取 (解決畫面未更新)"):
            st.cache_data.clear()
            st.rerun()
            
        st.divider()
        
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.write("🙈 **隱藏分享者 Tab** (不刪除歷史數據)")
            all_t_db = db_session.query(Tipster).filter_by(is_active=True).all()
            hide_target = st.selectbox("選擇要隱藏的分享者", [t.name for t in all_t_db])
            if st.button("執行隱藏"):
                t_obj = db_session.query(Tipster).filter_by(name=hide_target).first()
                if t_obj:
                    t_obj.is_active = False
                    db_session.commit()
                    st.success("隱藏成功，重新載入中...")
                    st.rerun()
                    
        with c_m2:
            st.write("🧹 **清理下拉選單垃圾資料**")
            all_tips = db_session.query(Tip).all()
            raw_teams = list(set([t.home_team for t in all_tips] + [t.away_team for t in all_tips]))
            raw_tours = list(set([t.tournament for t in all_tips if t.tournament]))
            
            hidden_opts = db_session.query(HiddenOption).all()
            hidden_teams = [h.option_name for h in hidden_opts if h.option_type == 'team']
            hidden_tours = [h.option_name for h in hidden_opts if h.option_type == 'tournament']
            
            clean_teams = [t for t in raw_teams if t and t not in hidden_teams]
            clean_tours = [t for t in raw_tours if t and t not in hidden_tours]
            
            hide_team_opts = st.multiselect("隱藏特定的隊伍名稱 (不再出現於下拉)", clean_teams)
            hide_tour_opts = st.multiselect("隱藏特定的賽事名稱 (不再出現於下拉)", clean_tours)
            
            if st.button("確認加入黑名單"):
                for tm in hide_team_opts: db_session.add(HiddenOption(option_type='team', option_name=tm))
                for tr in hide_tour_opts: db_session.add(HiddenOption(option_type='tournament', option_name=tr))
                db_session.commit()
                st.success("選單已清理！")
                st.rerun()

    # ----------------------------------------
    # [Tab] 動態 Tipster Tabs
    # ----------------------------------------
    # 預先抓取過濾後的下拉名單
    all_tips_global = db_session.query(Tip).all()
    hidden_opts_global = db_session.query(HiddenOption).all()
    h_teams_g = [h.option_name for h in hidden_opts_global if h.option_type == 'team']
    h_tours_g = [h.option_name for h in hidden_opts_global if h.option_type == 'tournament']
    
    history_teams = [x for x in list(set([t.home_team for t in all_tips_global] + [t.away_team for t in all_tips_global])) if x and x not in h_teams_g]
    history_tours = [x for x in list(set([t.tournament for t in all_tips_global if t.tournament])) if x and x not in h_tours_g]

    for i, tipster in enumerate(tipsters):
        with tabs[i]:
            st.header(f"分享者：{tipster.name} 的盤口建議")
            
            # --- 撤回並重填預設值注入 ---
            preset = st.session_state.pop(f"edit_preset_{tipster.id}", None)
            if preset:
                st.session_state[f"cat_{tipster.id}"] = preset['category']
                if preset['tournament'] in history_tours:
                    st.session_state[f"tour_{tipster.id}"] = preset['tournament']
                    st.session_state[f"tour_new_{tipster.id}"] = ""
                else:
                    st.session_state[f"tour_{tipster.id}"] = ""
                    st.session_state[f"tour_new_{tipster.id}"] = preset['tournament']
                    
                if preset['home_team'] in history_teams:
                    st.session_state[f"ht_{tipster.id}"] = preset['home_team']
                    st.session_state[f"ht_new_{tipster.id}"] = ""
                else:
                    st.session_state[f"ht_{tipster.id}"] = ""
                    st.session_state[f"ht_new_{tipster.id}"] = preset['home_team']
                    
                if preset['away_team'] in history_teams:
                    st.session_state[f"at_{tipster.id}"] = preset['away_team']
                    st.session_state[f"at_new_{tipster.id}"] = ""
                else:
                    st.session_state[f"at_{tipster.id}"] = ""
                    st.session_state[f"at_new_{tipster.id}"] = preset['away_team']
                    
                st.session_state[f"mk_{tipster.id}"] = preset['market_type']
                st.session_state[f"line_{tipster.id}_{preset['market_type']}"] = preset['line']
                st.session_state[f"sel_{tipster.id}"] = preset['selection']
                st.session_state["odds1"] = preset['odds']
                st.session_state["odds2"] = calculate_linked_odds(preset['odds'])

            colL, colR = st.columns([1.5, 1.2])
            
            with colL:
                st.subheader("📝 紀錄新建議")
                cat = st.selectbox("賽事分類", ["國內聯賽", "國內盃賽", "國際聯賽", "國際盃賽", "友誼賽"], key=f"cat_{tipster.id}")
                
                tour = st.selectbox("賽事名稱", [""] + history_tours, key=f"tour_{tipster.id}")
                if not tour: tour = st.text_input("或輸入新賽事名稱", key=f"tour_new_{tipster.id}")
                
                c1, c2 = st.columns(2)
                with c1: 
                    h_team = st.selectbox("主隊名稱", [""] + history_teams, key=f"ht_{tipster.id}")
                    if not h_team: h_team = st.text_input("或輸入新主隊", key=f"ht_new_{tipster.id}")
                with c2: 
                    a_team = st.selectbox("客隊名稱", [""] + history_teams, key=f"at_{tipster.id}")
                    if not a_team: a_team = st.text_input("或輸入新客隊", key=f"at_new_{tipster.id}")

                market = st.selectbox("盤口", ["讓球", "半場讓球", "入球大小", "半場入球大小", "角球大小", "半場角球大小", "讓角", "半場讓角"], key=f"mk_{tipster.id}")
                
                if market in ["讓球", "半場讓球", "讓角", "半場讓角"]: default_line = 0.0
                elif market == "入球大小": default_line = 2.5
                elif market == "半場入球大小": default_line = 1.5
                elif market == "角球大小": default_line = 9.5
                elif market == "半場角球大小": default_line = 4.5
                else: default_line = 0.0
                
                line = st.number_input("盤口線", value=default_line, step=0.25, key=f"line_{tipster.id}_{market}")
                selection = st.radio("分享者的選擇", ["主", "客", "大", "小"], horizontal=True, key=f"sel_{tipster.id}")
                
                c_o1, c_o2, c_o3 = st.columns(3)
                with c_o1:
                    st.number_input(f"賠率 ({selection})", min_value=1.01, step=0.01, key="odds1", on_change=update_odds1)
                with c_o2:
                    opp_label = get_opposite_selection(selection)
                    st.number_input(f"賠率 ({opp_label})", min_value=1.01, step=0.01, key="odds2")
                with c_o3:
                    margin = calculate_actual_margin(st.session_state.odds1, st.session_state.odds2)
                    st.info(f"盤口抽水: **{margin:.4f}**")

            with colR:
                st.subheader("🤖 AI 預測與系統學習")
                sys_bankroll = get_avail_bankroll('System', None, db_session)
                
                if st.button("🧠 執行 AI 深度分析與系統策略", key=f"run_ai_{tipster.id}", use_container_width=True):
                    res = analyze_system_edge(tipster.id, st.session_state.odds1, st.session_state.odds2, line, sys_bankroll, db_session)
                    st.session_state[f"ai_res_{tipster.id}"] = res
                
                ai_data = st.session_state.get(f"ai_res_{tipster.id}")
                
                # 預設變數
                sys_rec_stake, sys_choice, sys_odds = 0.0, selection, st.session_state.odds1 
                
                if ai_data:
                    sys_rec_stake, p_win_t, p_win_o, ev_t, ev_o, sys_direction, ai_status = ai_data
                    
                    st.caption(f"模型狀態: {ai_status}")
                    
                    # 顯示兩邊勝率與期望值
                    cc1, cc2 = st.columns(2)
                    cc1.metric(f"分享者 ({selection}) 勝率", f"{p_win_t*100:.1f}%", f"EV: {ev_t:.2f}")
                    opp_sel = get_opposite_selection(selection)
                    cc2.metric(f"對立面 ({opp_sel}) 勝率", f"{p_win_o*100:.1f}%", f"EV: {ev_o:.2f}")
                    
                    if sys_direction == 'Opposite':
                        st.warning(f"⚠️ **系統判定**：對立面 ({opp_sel}) 的期望值更高！建議系統反下。")
                        sys_choice = opp_sel
                        sys_odds = st.session_state.odds2
                    else:
                        st.success(f"✅ **系統判定**：支持分享者 ({selection}) 具備正期望值！")
                        sys_choice = selection
                        sys_odds = st.session_state.odds1
                        
                    st.metric("🤖 系統建議下注額 (Kelly)", f"${sys_rec_stake:.2f}")
                else:
                    st.info("請點擊上方按鈕執行分析。")

            st.divider()
            st.write("🛒 **多重下注確認面板 (可同時下多方帳戶)**")
            c_s1, c_s2, c_s3 = st.columns(3)
            with c_s1:
                do_tipster = st.checkbox(f"✅ 紀錄 {tipster.name} 注單", value=True)
                stake_tipster = st.number_input("分享者下注額", value=100.0, step=10.0, key=f"stk_t_{tipster.id}")
            with c_s2:
                do_user = st.checkbox("👤 用家跟投 (User)", value=False)
                stake_user = st.number_input("用家下注額", value=100.0, step=10.0, key=f"stk_u_{tipster.id}")
            with c_s3:
                do_sys = st.checkbox(f"🤖 系統智能下注 ({sys_choice})", value=bool(ai_data and sys_rec_stake > 0))
                stake_sys = st.number_input("系統下注額", value=float(sys_rec_stake), step=10.0, key=f"stk_s_{tipster.id}")
            
            if st.button("🚀 確認提交注單", key=f"submit_{tipster.id}", type="primary", use_container_width=True):
                if not tour or not h_team or not a_team:
                    st.error("賽事、主隊、客隊名稱不能為空")
                else:
                    common_kwargs = {
                        'category': cat, 'tournament': tour, 'home_team': h_team, 'away_team': a_team,
                        'market_type': market, 'line': line
                    }
                    if do_tipster:
                        db_session.add(Tip(tipster_id=tipster.id, owner_type='Tipster', selection=selection, odds=st.session_state.odds1, stake=stake_tipster, **common_kwargs))
                    if do_user:
                        db_session.add(Tip(tipster_id=tipster.id, owner_type='User', selection=selection, odds=st.session_state.odds1, stake=stake_user, **common_kwargs))
                    if do_sys:
                        db_session.add(Tip(tipster_id=tipster.id, owner_type='System', selection=sys_choice, odds=sys_odds, stake=stake_sys, **common_kwargs))
                    
                    db_session.commit()
                    st.success("紀錄成功！")
                    st.rerun()

            # --- 撤回並重填最新一筆 (僅針對 Tipster 的注單) ---
            latest_tip = db_session.query(Tip).filter_by(tipster_id=tipster.id, owner_type='Tipster', is_deleted=False).order_by(Tip.id.desc()).first()
            if latest_tip:
                if st.button("🔙 發現錯漏？撤回並重填最新一筆注單", key=f"edit_latest_{tipster.id}"):
                    st.session_state[f"edit_preset_{tipster.id}"] = {
                        'category': latest_tip.category, 'tournament': latest_tip.tournament,
                        'home_team': latest_tip.home_team, 'away_team': latest_tip.away_team,
                        'market_type': latest_tip.market_type, 'line': latest_tip.line,
                        'selection': latest_tip.selection, 'odds': latest_tip.odds
                    }
                    db_session.delete(latest_tip)
                    db_session.commit()
                    st.rerun()

    # ----------------------------------------
    # [Tab] 資金總覽與結算
    # ----------------------------------------
    with tabs[-2]:
        st.header("資金總覽與賽果結算")
        
        with st.expander("📝 存取資金操作 (注入/提取本金)"):
            c_f1, c_f2 = st.columns(2)
            with c_f1:
                acc_type_opt = st.selectbox("對象", ["👑 系統專屬資金", "👤 用家真實資金", "👥 分享者虛擬本金"])
                target_tipster = None
                if acc_type_opt == "👥 分享者虛擬本金":
                    target_tipster = st.selectbox("選擇分享者", tipster_names)
            with c_f2:    
                action = st.radio("操作", ["存入 (紅色負數：投入本金)", "提取 (綠色正數：回收本金)"], horizontal=True)
                amt_input = st.number_input("金額", min_value=0.0, step=100.0)
                desc = st.text_input("備註 (選填)")
            
            if st.button("提交資金變動"):
                if amt_input > 0:
                    actual_amt = round(-amt_input if "存入" in action else amt_input, 2)
                    
                    if "系統" in acc_type_opt: db_acc = 'System'
                    elif "用家" in acc_type_opt: db_acc = 'User'
                    else: db_acc = 'Tipster'
                    
                    t_id = tipster_dict.get(target_tipster) if target_tipster else None
                    
                    db_session.add(BankrollLedger(account_type=db_acc, tipster_id=t_id, amount=actual_amt, description=desc))
                    db_session.commit()
                    st.success("資金流水已成功紀錄！")
                    st.rerun()

        st.subheader("📊 資金總覽 (表列方式)")
        summary_data = []
        for role, name, t_id in [('System', '👑 系統 (System)', None), ('User', '👤 用家 (User)', None)] + [('Tipster', f'👥 {t.name}', t.id) for t in tipsters]:
            ledgers = db_session.query(BankrollLedger).filter_by(account_type=role)
            if t_id: ledgers = ledgers.filter_by(tipster_id=t_id)
            ledgers = ledgers.all()
            
            tips_q = db_session.query(Tip).filter(Tip.status != 'Open', Tip.is_deleted == False, Tip.owner_type == role)
            if t_id: tips_q = tips_q.filter_by(tipster_id=t_id)
            tips = tips_q.all()
            
            dep = sum(l.amount for l in ledgers if l.amount < 0)
            wit = sum(l.amount for l in ledgers if l.amount > 0)
            prof = sum(t.profit for t in tips)
            
            summary_data.append({
                '帳戶名稱': name, '總存入本金': dep, '總提取本金': wit,
                '累計總盈虧': prof, '當前可用資金': abs(dep) - wit + prof
            })
            
        df_summary = pd.DataFrame(summary_data)
        df_summary_cols = ['總存入本金', '總提取本金', '累計總盈虧', '當前可用資金']
        for c in df_summary_cols: df_summary[c] = df_summary[c].astype(float)
        st.dataframe(df_summary.style.map(style_financials, subset=df_summary_cols).format({c: "{:.2f}" for c in df_summary_cols}), use_container_width=True)
        st.divider()

        st.subheader("🏁 待結算賽事 (群組批量結算)")
        open_tips = db_session.query(Tip).filter_by(status='Open', is_deleted=False).all()
        
        if not open_tips:
            st.info("目前無待結算注單。")
        else:
            # 依照賽事特徵群組化注單 (系統、用家、分享者的同場賽事會綁在一起結算)
            matches = {}
            for t in open_tips:
                m_key = f"{t.tournament} | {t.home_team} vs {t.away_team}"
                if m_key not in matches: matches[m_key] = []
                matches[m_key].append(t)
                
            for m_key, m_tips in matches.items():
                with st.expander(f"📍 {m_key} (共包含 {len(m_tips)} 張注單)"):
                    st.write("**包含的注單細節：**")
                    for t in m_tips:
                        t_name_display = db_session.query(Tipster).get(t.tipster_id).name if t.tipster_id else "N/A"
                        st.caption(f"- [{t.owner_type} - {t_name_display}] 盤口: {t.market_type}({t.line}) | 選項: **{t.selection}** | 賠率: {t.odds} | 投注: ${t.stake}")
                    
                    st.divider()
                    c1, c2, c3, c4 = st.columns(4)
                    h_ht_sc = c1.number_input("主隊半場進球", min_value=0, step=1, key=f"h_ht_sc_{m_key}")
                    a_ht_sc = c2.number_input("客隊半場進球", min_value=0, step=1, key=f"a_ht_sc_{m_key}")
                    h_sc = c3.number_input("主隊全場進球", min_value=0, step=1, key=f"h_sc_{m_key}")
                    a_sc = c4.number_input("客隊全場進球", min_value=0, step=1, key=f"a_sc_{m_key}")

                    c5, c6, c7, c8 = st.columns(4)
                    h_ht_cor = c5.number_input("主隊半場角球", min_value=0, step=1, key=f"h_ht_cor_{m_key}")
                    a_ht_cor = c6.number_input("客隊半場角球", min_value=0, step=1, key=f"a_ht_cor_{m_key}")
                    h_cor = c7.number_input("主隊全場角球", min_value=0, step=1, key=f"h_cor_{m_key}")
                    a_cor = c8.number_input("客隊全場角球", min_value=0, step=1, key=f"a_cor_{m_key}")
                    
                    if st.button("✅ 結算此賽事所有注單", key=f"set_{m_key}", type="primary"):
                        for t in m_tips:
                            status, unit_profit = settle_asian_handicap(
                                t.market_type, t.line, t.selection, 
                                h_sc, a_sc, h_ht_sc, a_ht_sc, h_cor, a_cor, h_ht_cor, a_ht_cor, t.odds
                            )
                            profit = round(t.stake * unit_profit, 2)
                            payout = round(t.stake + profit if unit_profit >= -0.5 else 0.0, 2)
                            
                            old_data = {
                                'status': t.status, 'is_deleted': t.is_deleted, 
                                'unit_profit': t.unit_profit, 'profit': t.profit, 'payout': t.payout,
                                'home_score': t.home_score, 'away_score': t.away_score,
                                'home_ht_score': t.home_ht_score, 'away_ht_score': t.away_ht_score,
                                'home_corners': t.home_corners, 'away_corners': t.away_corners,
                                'home_ht_corners': t.home_ht_corners, 'away_ht_corners': t.away_ht_corners
                            }
                            db_session.add(ActionLog(action_type='SETTLE', record_id=t.id, table_name='tips', old_data=json.dumps(old_data)))

                            t.home_score, t.away_score, t.home_ht_score, t.away_ht_score = h_sc, a_sc, h_ht_sc, a_ht_sc
                            t.home_corners, t.away_corners, t.home_ht_corners, t.away_ht_corners = h_cor, a_cor, h_ht_cor, a_ht_cor
                            t.status, t.unit_profit, t.profit, t.payout = status, unit_profit, profit, payout
                        
                        db_session.commit()
                        st.success(f"賽事結算完成！")
                        st.rerun()

    db_session.close()

if __name__ == "__main__":
    main()
