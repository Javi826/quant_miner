// dashboard.js===========================================================================
// BOT_trading Dashboard JavaScript
// ===========================================================================

const COLORS = {
    purple: '#6d28d9',
    green: '#3fb950',
    healthy: '#58a6ff',
    warning: '#f0883e',   
    danger: '#f85149',    
    yellow: '#d29922',
    red: '#f85149',
    textPrimary: '#c9d1d9',
    textSecondary: '#8b949e',
    gridDark: '#21262d',
    borderYellow: '#facc15',
    white: '#ffffff',
    blue: '#58a6ff',
    equityPositive: '#3fb950',
    equityNegative: '#f85149',
    drawdownRed: '#f85149',
    drawdownRedAlpha: 'rgba(248, 81, 73, 0.1)'
};

const METRIC_THRESHOLDS = {
    profitFactor: { excellent: 2.0, good: 1.5, acceptable: 1.0 },
    sharpeRatio: { excellent: 2.0, good: 1.5, acceptable: 1.0 },
    rSquared: { excellent: 0.9, good: 0.7, acceptable: 0.5 }
};

const CHART_DEFAULTS = {
    fontSize: { title: 20, axis: 16 },
    gridColor: COLORS.gridDark,
    borderColor: COLORS.borderYellow,
    borderWidth: 1,
    textColor: COLORS.white,
    titleColor: COLORS.textPrimary
};

let SLIPPAGE_THRESHOLDS = {
    warning: 0.2,   // Fallback si falla el fetch
    critical: 0.3
};
async function loadQualityThresholds() {
    try {
        const res = await fetch('/api/quality/thresholds');
        const data = await res.json();
        if (data.success) {
            SLIPPAGE_THRESHOLDS.warning = data.thresholds.slippage_warning_pct;
            SLIPPAGE_THRESHOLDS.critical = data.thresholds.slippage_critical_pct;
        }
    } catch (error) {
        console.error('Error loading quality thresholds:', error);
    }
}
function getMetricColor(value, withGlow = false) {
    const thresholds = METRIC_THRESHOLDS.profitFactor;
    if (value >= thresholds.excellent) {
        return { color: COLORS.purple, shadow: withGlow ? '0 0 10px rgba(109, 40, 217, 0.8)' : 'none' };
    } else if (value >= thresholds.good) {
        return { color: COLORS.green, shadow: 'none' };
    } else if (value >= thresholds.acceptable) {
        return { color: COLORS.yellow, shadow: 'none' };
    } else {
        return { color: COLORS.red, shadow: 'none' };
    }
}

function getRSquaredColor(value) {
    if (value >= 0.9) return { color: COLORS.purple, shadow: '0 0 10px rgba(109, 40, 217, 0.8)' };
    if (value >= 0.7) return { color: COLORS.green, shadow: 'none' };
    if (value >= 0.5) return { color: COLORS.yellow, shadow: 'none' };
    return { color: COLORS.red, shadow: 'none' };
}

function getPositiveNegativeColor(value) {
    return value >= 0 ? COLORS.green : COLORS.red;
}

function applyMetricColor(element, value, type = 'profitFactor') {
    if (type === 'profitFactor' || type === 'sharpe') {
        const { color, shadow } = getMetricColor(value, true);
        element.style.color = color;
        element.style.textShadow = shadow;
    } else if (type === 'rSquared') {
        const { color, shadow } = getRSquaredColor(value);
        element.style.color = color;
        element.style.textShadow = shadow;
    } else if (type === 'positiveNegative') {
        element.style.color = getPositiveNegativeColor(value);
    }
}

function getBaseChartConfig(title, reverseY = false) {
    const config = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            title: { 
                display: true, 
                text: title,
                color: CHART_DEFAULTS.titleColor,
                font: { size: CHART_DEFAULTS.fontSize.title, weight: 'bold' }
            }
        },
        scales: {
            x: { 
                ticks: { color: CHART_DEFAULTS.textColor, font: { size: CHART_DEFAULTS.fontSize.axis } }, 
                grid: { 
                    color: CHART_DEFAULTS.gridColor,
                    drawBorder: true,
                    borderColor: CHART_DEFAULTS.borderColor,
                    borderWidth: CHART_DEFAULTS.borderWidth
                } 
            },
            y: { 
                ticks: { 
                    color: CHART_DEFAULTS.textColor,
                    font: { size: CHART_DEFAULTS.fontSize.axis },
                    callback: function(value) { return value.toFixed(1) + '%'; }
                }, 
                grid: { 
                    color: CHART_DEFAULTS.gridColor,
                    drawBorder: true,
                    borderColor: CHART_DEFAULTS.borderColor,
                    borderWidth: CHART_DEFAULTS.borderWidth
                }
            }
        }
    };
    if (reverseY) config.scales.y.reverse = true;
    return config;
}

function clearAnalysisDates() {
    document.getElementById('analysis-date-from').value = '';
    document.getElementById('analysis-date-to').value = '';
}

function getAnalysisDateParams() {
    const dateFrom = document.getElementById('analysis-date-from').value;
    const dateTo = document.getElementById('analysis-date-to').value;
    let params = '';
    if (dateFrom) params += '&date_from=' + dateFrom;
    if (dateTo) params += '&date_to=' + dateTo;
    return params;
}

// Date filter helper functions
function clearCurvesDates() {
    document.getElementById('curves-date-from').value = '';
    document.getElementById('curves-date-to').value = '';
}

function getCurvesDateParams() {
    const dateFrom = document.getElementById('curves-date-from').value;
    const dateTo = document.getElementById('curves-date-to').value;
    let params = '';
    if (dateFrom) params += '&date_from=' + dateFrom;
    if (dateTo) params += '&date_to=' + dateTo;
    return params;
}

async function waitForBackend() {
    const maxAttempts = 30;
    let attempts = 0;
    const splash = document.getElementById('loading-splash');
    const statusText = document.getElementById('loading-status');
    const attemptCount = document.getElementById('attempt-count');
    const errorBox = document.getElementById('loading-error');
    
    splash.classList.remove('hidden');
    splash.style.opacity = '1';
    splash.style.display = 'flex';
    errorBox.classList.remove('visible');
    
    while (attempts < maxAttempts) {
        attempts++;
        attemptCount.textContent = attempts;
        
        try {
            const response = await fetch('/api/status', { 
                method: 'GET',
                cache: 'no-cache',
                headers: { 'Accept': 'application/json', 'Cache-Control': 'no-cache' }
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.status === 'running' || data.account) {
                    statusText.innerHTML = '✅ Connected successfully!';
                    statusText.style.color = '#4ade80';
                    await new Promise(r => setTimeout(r, 500));
                    splash.style.transition = 'opacity 0.3s ease';
                    splash.style.opacity = '0';
                    await new Promise(r => setTimeout(r, 300));
                    splash.classList.add('hidden');
                    splash.style.display = 'none';
                    return true;
                }
            }
        } catch (error) {}
        await new Promise(r => setTimeout(r, 1000));
    }
    
    errorBox.classList.add('visible');
    setTimeout(() => {
        splash.style.transition = 'opacity 0.5s ease';
        splash.style.opacity = '0';
        setTimeout(() => {
            splash.classList.add('hidden');
            splash.style.display = 'none';
        }, 500);
    }, 5000);
    return false;
}

let autoScroll = true;
let isLoadingData = false;
let isLoadingLogs = false;
let currentPositionsView = 'compact';
let cachedPositions = [];
let currentTab = 'positions';
let equityChart = null;
let drawdownChart = null;
let allStrategiesList = [];
let isStoppingBot = false;

// ═══════════════════════════════════════════════════════════════════════════
// POSITION SORT FEATURE (NEW)
// ═══════════════════════════════════════════════════════════════════════════

let positionSortBy = 'tp';

function sortPositionsBy(type) {
    positionSortBy = type;
    
    // Update button states
    const buttons = document.querySelectorAll('#sort-buttons .view-btn');
    buttons.forEach(btn => btn.classList.remove('active'));
    
    // Find and activate the clicked button
    buttons.forEach(btn => {
        if (btn.textContent.trim().toUpperCase() === type.toUpperCase()) {
            btn.classList.add('active');
        }
    });
    
    renderPositions(cachedPositions);
}

// ═══════════════════════════════════════════════════════════════════════════
// END POSITION SORT FEATURE
// ═══════════════════════════════════════════════════════════════════════════

async function stopBot() {
    if (isStoppingBot) return;
    if (!confirm('WARNING: STOP TRADING BOT?\n\nThis will terminate the bot process.\n\nAre you sure?')) return;
    isStoppingBot = true;
    
    const overlay = document.getElementById('stop-overlay');
    const statusDiv = document.getElementById('stop-status');
    statusDiv.innerHTML = '';
    overlay.classList.add('active');
    
    addStopLog('Initializing stop sequence...');
    addStopLog('Stop signal requested');
    
    try {
        const stopResponse = await fetch('/api/bot/stop', { method: 'POST' });
        if (!stopResponse.ok) throw new Error('Failed to send stop signal');
        
        const stopData = await stopResponse.json();
        addStopLog('Stop signal sent (SIGTERM) to PID ' + stopData.pid, 'success');
        
        let attempts = 0;
        const maxAttempts = 30;
        let botStoppedConfirmed = false;
        
        const verifyInterval = setInterval(async () => {
            attempts++;
            addStopLog('Verifying shutdown... (' + attempts + '/' + maxAttempts + ')');
            
            try {
                const verifyResponse = await fetch('/api/bot/verify-stopped', {
                    method: 'GET',
                    cache: 'no-cache'
                });
                
                if (verifyResponse.ok) {
                    const status = await verifyResponse.json();
                    if (!status.running) {
                        clearInterval(verifyInterval);
                        botStoppedConfirmed = true;
                        addStopLog('Process verified as terminated', 'success');
                        addStopLog('✅ BOT STOPPED SUCCESSFULLY', 'success');
                        document.getElementById('stop-title').textContent = '✅ BOT STOPPED';
                        document.getElementById('stop-title').style.color = COLORS.green;
                        return;
                    }
                }
            } catch (error) {
                clearInterval(verifyInterval);
                botStoppedConfirmed = true;
                addStopLog('Backend stopped responding', 'success');
                addStopLog('Flask server terminated (expected)', 'success');
                addStopLog('✅ BOT STOPPED SUCCESSFULLY', 'success');
                document.getElementById('stop-title').textContent = '✅ BOT STOPPED';
                document.getElementById('stop-title').style.color = COLORS.green;
                return;
            }
            
            if (attempts >= maxAttempts && !botStoppedConfirmed) {
                clearInterval(verifyInterval);
                addStopLog('⚠️ VERIFICATION TIMEOUT', 'warning');
                addStopLog('Bot may still be running. Check manually.', 'warning');
                document.getElementById('stop-title').textContent = '⚠️ TIMEOUT';
                document.getElementById('stop-title').style.color = COLORS.yellow;
            }
        }, 1000);
        
    } catch (error) {
        if (error.message.includes('fetch') || error.name === 'TypeError') {
            addStopLog('⚠️ Backend not responding', 'warning');
            addStopLog('Bot may have already stopped', 'warning');
        } else {
            addStopLog('❌ ERROR: ' + error.message, 'error');
        }
    }
}

function addStopLog(message, className = '') {
    const statusDiv = document.getElementById('stop-status');
    const line = document.createElement('div');
    line.className = 'stop-status-line' + (className ? ' ' + className : '');
    line.textContent = message;
    statusDiv.appendChild(line);
    statusDiv.scrollTop = statusDiv.scrollHeight;
}

function switchTab(tabName) {
    currentTab = tabName;
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    document.getElementById('tab-' + tabName).classList.add('active');
    
    if (tabName === 'analysis') loadStrategyAnalysis();
    if (tabName === 'config') loadBotConfig();
    if (tabName === 'equity') loadEquityTab();
    if (tabName === 'risk') loadRiskTab();
    if (tabName === 'quality') loadQualityTab();
}

function switchEquitySubTab(subTabName) {
    document.querySelectorAll('#tab-equity .tabs-container .tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    document.querySelectorAll('#tab-equity .tab-content').forEach(content => content.classList.remove('active'));
    document.getElementById('equity-subtab-' + subTabName).classList.add('active');
    
    if (subTabName === 'weekday') loadWeekDayAnalysis();
    if (subTabName === 'period') initPeriodTab();
}

function getLogClass(line) {
    if (line.includes('TP for')) return 'tp-hit';
    if (line.includes('SL for')) return 'sl-hit';
    if (line.includes('TIMEOUT')) return 'info';
    if (!line.includes('SHORTS') && !line.includes('LONGS')) {
        if (line.includes('SHORT') || line.includes('LONG')) return 'info';
    }
    if (line.includes('Error')) return 'error';
    if (line.includes('WAR')) return 'warning';
    return 'default';
}


async function loadLogs() {
    if (isLoadingLogs) return;
    isLoadingLogs = true;
    try {
        const res = await fetch('/api/logs/stream');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        
        const logsContent = document.getElementById('logs-content');
        if (!logsContent) return;
        
        // REEMPLAZAR TODO el contenido
        logsContent.innerHTML = '';
        
        if (data.logs && data.logs.length > 0) {
            const fragment = document.createDocumentFragment();
            data.logs.forEach(log => {
                const logLine = document.createElement('div');
                logLine.className = 'log-line ' + getLogClass(log);
                logLine.textContent = log;
                fragment.appendChild(logLine);
            });
            logsContent.appendChild(fragment);
            
            // Auto-scroll al final
            if (autoScroll) {
                logsContent.scrollTop = logsContent.scrollHeight;
            }
        }
    } catch (error) {
        console.error('Error loading logs:', error);
    } finally {
        isLoadingLogs = false;
    }
}

function setPositionsView(view) {
    currentPositionsView = view;
    document.querySelectorAll('.view-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    if (view !== 'symbols') {
        const headerSpan = document.querySelector('#tab-positions .content-section h2 span');
        if (headerSpan) {
            headerSpan.textContent = 'Active Positions';
        }
    }
    
    renderPositions(cachedPositions);
    localStorage.setItem('positionsView', view);
}

function renderPositions(positions) {
    cachedPositions = positions;
    const container = document.getElementById('positions-container');
    
    if (!positions || positions.length === 0) {
        container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No active positions</div>';
        return;
    }
    
    if (currentPositionsView === 'compact') {
        renderCompactView(container, positions);
    } else if (currentPositionsView === 'detailed') {
        renderDetailedView(container, positions);
    } else if (currentPositionsView === 'symbols') {
        renderSymbolsView(container, positions);
    }
}

function renderCompactView(container, positions) {
    const groupedByStrategy = {};
    positions.forEach(pos => {
        if (!groupedByStrategy[pos.strategy]) {
            groupedByStrategy[pos.strategy] = {
                positions: [],
                totalPnl: 0,
                direction: pos.direction,
                opened_at: pos.opened_at,
                candles: pos.candles,
                max_candles: pos.max_candles
            };
        }
        groupedByStrategy[pos.strategy].positions.push(pos);
        groupedByStrategy[pos.strategy].totalPnl += (pos.current_pnl || 0);
    });
    
    let allEntries = [];
    
    if (allStrategiesList && allStrategiesList.length > 0) {
        const allActiveDeprecating = allStrategiesList.filter(s => 
            s.status === 'ACTIVE' || s.status === 'DEPRECATING'
        );
        
        allActiveDeprecating.forEach(strat => {
            if (groupedByStrategy[strat.id]) {
                allEntries.push([strat.id, groupedByStrategy[strat.id]]);
            } else {
                allEntries.push([strat.id, {
                    positions: [],
                    totalPnl: 0,
                    direction: strat.direction,
                    opened_at: null,
                    candles: 0,
                    max_candles: 50,
                    isEmpty: true
                }]);
            }
        });
    } else {
        allEntries = Object.entries(groupedByStrategy);
    }
    
    const sortedEntries = allEntries.sort((a, b) => a[0].localeCompare(b[0]));
    
    if (sortedEntries.length === 0) {
        container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No active positions</div>';
        return;
    }
    
    const html = '<table><thead><tr><th>#</th><th>Strategy</th><th>Side</th><th>Opened</th><th style="text-align: right;">Candles</th><th style="text-align: center;">#pos</th><th>PnL</th></tr></thead><tbody>' +
        sortedEntries.map(([strategyId, data], index) => {
            const pnl = data.totalPnl;
            const pnlClass = pnl >= 0 ? 'direction-long' : 'direction-short';
            const num = String(index + 1).padStart(2, '0');
            
            let openedDateStr = '-';
            if (data.opened_at) {
                try {
                    const date = new Date(data.opened_at);
                    openedDateStr = date.toISOString().split('T')[0];
                } catch {
                    openedDateStr = String(data.opened_at).substring(0, 10);
                }
            }
            
            if (data.isEmpty) {
                return '<tr><td style="color: #8b949e; font-weight: 600;">' + num + '</td><td>' + strategyId + '</td><td class="direction-' + data.direction.toLowerCase() + '">' + data.direction.toUpperCase() + '</td><td>-</td><td style="text-align: right;">-</td><td style="text-align: center; color: #f85149; font-weight: 600;">0</td><td>-</td></tr>';
            }
            
            return '<tr><td style="color: #8b949e; font-weight: 600;">' + num + '</td><td>' + strategyId + '</td><td class="direction-' + data.direction.toLowerCase() + '">' + data.direction.toUpperCase() + '</td><td>' + openedDateStr + '</td><td style="text-align: right;">' + (data.candles || 0) + '/' + (data.max_candles || 50) + '</td><td style="text-align: center; color: #58a6ff; font-weight: 600;">' + data.positions.length + '</td><td class="' + pnlClass + '">' + (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2) + '</td></tr>';
        }).join('') +
        '</tbody></table>';
    container.innerHTML = html;
}

function renderDetailedView(container, positions) {
    // MODIFIED: Use positionSortBy to determine sort field
    const positionsWithDelta = positions.map(pos => {
        const currentPrice = parseFloat(pos.current_price || pos.entry_price);
        const tp = parseFloat(pos.tp);
        const sl = parseFloat(pos.sl);
        
        let deltaTp, deltaSl;
        if (pos.direction.toLowerCase() === 'long') {
            deltaTp = ((tp - currentPrice) / currentPrice * 100);
            deltaSl = ((currentPrice - sl) / currentPrice * 100);
        } else {
            deltaTp = ((currentPrice - tp) / currentPrice * 100);
            deltaSl = ((sl - currentPrice) / currentPrice * 100);
        }
        
        return { ...pos, _deltaTp: deltaTp, _deltaSl: deltaSl };
    });
    
    // MODIFIED: Sort based on positionSortBy variable
    const sortedPositions = positionsWithDelta.sort((a, b) => {
        if (positionSortBy === 'tp') {
            return a._deltaTp - b._deltaTp;
        } else {
            return a._deltaSl - b._deltaSl;
        }
    });
    
    // Determine active button class
    const tpBtnClass = positionSortBy === 'tp' ? 'view-btn active' : 'view-btn';
    const slBtnClass = positionSortBy === 'sl' ? 'view-btn active' : 'view-btn';
    
    const html = '<table><thead><tr>' +
        '<th>Strategy</th>' +
        '<th>Symbol</th>' +
        '<th>Side</th>' +
        '<th>Entry</th>' +
        '<th>Current</th>' +
        '<th>Amount</th>' +
        '<th>TP (Δ%) <button class="' + tpBtnClass + '" onclick="sortPositionsBy(\'tp\')" style="margin-left: 8px; font-size: 11px; padding: 2px 8px;">TP</button></th>' +
        '<th>SL (Δ%) <button class="' + slBtnClass + '" onclick="sortPositionsBy(\'sl\')" style="margin-left: 8px; font-size: 11px; padding: 2px 8px;">SL</button></th>' +
        '<th>PnL</th>' +
        '<th style="text-align: right;">Candles</th>' +
        '</tr></thead><tbody>' +
        sortedPositions.map(pos => {
            const pnl = pos.current_pnl || 0;
            const pnlClass = pnl >= 0 ? 'direction-long' : 'direction-short';
            
            const currentPrice = pos.current_price;
            const tp = pos.tp;
            const sl = pos.sl;
            const entryPrice = pos.entry_price;
            const precision = pos.precision || 2;
            
            const deltaTp = pos.distance_to_tp_pct || 0;
            const deltaSl = pos.distance_to_sl_pct || 0;
            
            const deltaTpClass = Math.abs(deltaTp) < 1 ? 'delta-tp-close' : 'delta-tp';
            const deltaSlClass = Math.abs(deltaSl) < 1 ? 'delta-sl-close' : 'delta-sl';
            
            return '<tr>' +
                '<td>' + pos.strategy + '</td>' +
                '<td>' + pos.symbol + '</td>' +
                '<td class="direction-' + pos.direction.toLowerCase() + '">' + pos.direction.toUpperCase() + '</td>' +
                '<td>$' + entryPrice.toFixed(precision) + '</td>' +
                '<td style="color: #f0883e;">$' + currentPrice.toFixed(precision) + '</td>' +
                '<td>$' + parseFloat(pos.usdt_amount).toFixed(2) + '</td>' +
                '<td>$' + tp.toFixed(precision) + ' <span class="' + deltaTpClass + '">(Δ' + (deltaTp >= 0 ? '+' : '') + deltaTp.toFixed(2) + '%)</span></td>' +
                '<td>$' + sl.toFixed(precision) + ' <span class="' + deltaSlClass + '">(Δ' + (deltaSl >= 0 ? '+' : '') + deltaSl.toFixed(2) + '%)</span></td>' +
                '<td class="' + pnlClass + '">' + (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2) + '</td>' +
                '<td style="text-align: right;">' + (pos.candles || 0) + '/' + (pos.max_candles || 50) + '</td>' +
                '</tr>';
        }).join('') +
        '</tbody></table>';
    container.innerHTML = html;
}

function renderSymbolsView(container, positions) {
    const groupedBySymbolSide = {};
    positions.forEach(pos => {
        const key = pos.symbol + '_' + pos.direction.toUpperCase();
        
        if (!groupedBySymbolSide[key]) {
            groupedBySymbolSide[key] = {
                symbol: pos.symbol,
                side: pos.direction.toUpperCase(),
                totalSize: 0,
                totalPnl: 0,
                strategies: new Set()
            };
        }
        groupedBySymbolSide[key].totalSize += parseFloat(pos.size);
        groupedBySymbolSide[key].totalPnl += (pos.current_pnl || 0);
        
        const strategyNumber = extractNumberFromId(pos.strategy);
        groupedBySymbolSide[key].strategies.add(strategyNumber);
    });
    
    const sortedKeys = Object.keys(groupedBySymbolSide).sort((a, b) => {
        const dataA = groupedBySymbolSide[a];
        const dataB = groupedBySymbolSide[b];
        
        if (dataA.symbol !== dataB.symbol) {
            return dataA.symbol.localeCompare(dataB.symbol);
        }
        if (dataA.side === 'LONG' && dataB.side === 'SHORT') return -1;
        if (dataA.side === 'SHORT' && dataB.side === 'LONG') return 1;
        return 0;
    });
    
    if (sortedKeys.length === 0) {
        container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No active positions</div>';
        return;
    }
    
    let longCount = 0;
    let shortCount = 0;
    sortedKeys.forEach(key => {
        if (groupedBySymbolSide[key].side === 'LONG') longCount++;
        else if (groupedBySymbolSide[key].side === 'SHORT') shortCount++;
    });
    
    const headerSpan = document.querySelector('#tab-positions .content-section h2 span');
    if (headerSpan) {
        headerSpan.textContent = 'Active Positions - 🪙 ' + sortedKeys.length + ' Unique Positions (' + longCount + ' LONG / ' + shortCount + ' SHORT)';
    }
    
    const html = '<table><thead><tr>' +
        '<th>#</th>' +
        '<th>Symbol</th>' +
        '<th>Side</th>' +
        '<th>Total Size</th>' +
        '<th>Strategies</th>' +
        '<th>PnL</th>' +
        '</tr></thead><tbody>' +
        sortedKeys.map((key, index) => {
            const data = groupedBySymbolSide[key];
            const pnlClass = data.totalPnl >= 0 ? 'direction-long' : 'direction-short';
            const sideClass = data.side === 'LONG' ? 'direction-long' : 'direction-short';
            const num = String(index + 1).padStart(2, '0');
            
            const stratNumbers = Array.from(data.strategies).sort((a, b) => {
                const numA = parseInt(a) || 0;
                const numB = parseInt(b) || 0;
                return numA - numB;
            });
            const stratDisplay = '[' + stratNumbers.join(', ') + ']';
            
            return '<tr>' +
                '<td style="color: #8b949e; font-weight: 600;">' + num + '</td>' +
                '<td style="color: #58a6ff; font-weight: 600;">' + data.symbol + '</td>' +
                '<td class="' + sideClass + '">' + data.side + '</td>' +
                '<td>' + data.totalSize.toFixed(2) + '</td>' +
                '<td style="color: #c9d1d9;">' + stratDisplay + '</td>' +
                '<td class="' + pnlClass + '">' + (data.totalPnl >= 0 ? '+' : '') + '$' + data.totalPnl.toFixed(2) + '</td>' +
                '</tr>';
        }).join('') +
        '</tbody></table>';
    
    container.innerHTML = html;
}

async function loadStrategyAnalysis() {
    try {
        const dateParams = getAnalysisDateParams();
        const res = await fetch('/api/strategy-analysis?' + dateParams);
        const data = await res.json();
        const container = document.getElementById('analysis-container');
        
        if (!data || data.length === 0) {
            container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No data</div>';
            return;
        }
        
        const sortedData = data.sort((a, b) => a.Strategy.localeCompare(b.Strategy));
        
        const html = '<table><thead><tr><th>#</th><th>Strategy</th><th>First</th><th>Trades</th><th>Win %</th><th>Profit</th><th>Profit %</th><th>Total %</th><th>TP %</th><th>SL %</th><th>TIMEOUT %</th><th>OOM %</th><th>Avg Days</th></tr></thead><tbody>' +
            sortedData.map((s, index) => {
                const num = String(index + 1).padStart(2, '0');
                const profitClass = s.Total_profit >= 0 ? 'direction-long' : 'direction-short';
                return '<tr><td style="color: #8b949e; font-weight: 600;">' + num + '</td><td>' + s.Strategy + '</td><td>' + s.date_fo + '</td><td>' + s.Trades_num + '</td><td>' + s.Trades_pct.toFixed(1) + '%</td><td class="' + profitClass + '">' + (s.Total_profit >= 0 ? '+' : '') + '$' + s.Total_profit.toFixed(2) + '</td><td class="' + profitClass + '">' + (s.Profit_pct >= 0 ? '+' : '') + s.Profit_pct.toFixed(1) + '%</td><td>' + (s.Total_pct >= 0 ? '+' : '') + s.Total_pct.toFixed(1) + '%</td><td>' + s.TP_pct.toFixed(1) + '%</td><td>' + s.SL_pct.toFixed(1) + '%</td><td>' + s.TIMEOUT_pct.toFixed(1) + '%</td><td>' + s.OOM_pct.toFixed(1) + '%</td><td>' + s.Avg_days.toFixed(2) + '</td></tr>';
            }).join('') +
            '</tbody></table>';
        container.innerHTML = html;
    } catch (error) {
        console.error('Error:', error);
    }
}

async function loadBotConfig() {
    try {
        const res = await fetch('/api/bot-config');
        const data = await res.json();
        
        allStrategiesList = data.strategies || [];
        
        document.getElementById('config-account').textContent = data.account || '-';
        document.getElementById('config-capital').textContent = '$' + (data.initial_capital || 0).toLocaleString();
        document.getElementById('config-total-strat').textContent = data.stats.total || 0;
        document.getElementById('config-active-strat').textContent = data.stats.active || 0;
        document.getElementById('config-deprecating-strat').textContent = data.stats.deprecating || 0;
        document.getElementById('config-not-implemented-strat').textContent = data.stats.not_implemented || 0;
        
        const wsPublic = document.getElementById('ws-public');
        const wsPrivate = document.getElementById('ws-private');
        const wsAuth = document.getElementById('ws-auth');
        
        if (data.websocket_status.public_connected) {
            wsPublic.innerHTML = '<span class="ws-dot connected"></span><span>Connected</span>';
        } else {
            wsPublic.innerHTML = '<span class="ws-dot disconnected"></span><span>Disconnected</span>';
        }
        
        if (data.websocket_status.private_connected) {
            wsPrivate.innerHTML = '<span class="ws-dot connected"></span><span>Connected</span>';
        } else {
            wsPrivate.innerHTML = '<span class="ws-dot disconnected"></span><span>Disconnected</span>';
        }
        
        if (data.websocket_status.authenticated) {
            wsAuth.innerHTML = '<span class="ws-dot connected"></span><span>Authenticated</span>';
        } else {
            wsAuth.innerHTML = '<span class="ws-dot disconnected"></span><span>Not Authenticated</span>';
        }
        
        const strategiesBody = document.getElementById('strategies-body');
        if (!data.strategies || data.strategies.length === 0) {
            strategiesBody.innerHTML = '<tr><td colspan="15" style="text-align: center;">No strategies</td></tr>';
        } else {
            const sortedStrategies = data.strategies.sort((a, b) => a.id.localeCompare(b.id));
            
            // ✅ MODIFICACIÓN: Añadido 'dir_mode' a fixedKeys
            const fixedKeys = ['id', 'name', 'number', 'timeframe', 'status', 'symbols_count'];
            const commonKeys = ['tp_pct', 'sl_pct', 'order_amount', 'sell_after_ncandles'];
            const excludeKeys = new Set([...fixedKeys, ...commonKeys, 'direction', 'family_sizing', 'direction_mode', 'active',
                    'regime_trending', 'regime_ranging', 'regime_volatile',
                    'regime_trending_uptrend', 'regime_trending_dwtrend',
                    'regime_ranging_uptrend',  'regime_ranging_dwtrend',
                    'regime_volatile_uptrend', 'regime_volatile_dwtrend'
                ]);
            
            const extraParamKeys = new Set();
            sortedStrategies.forEach(strat => {
                Object.keys(strat).forEach(key => {
                    if (!excludeKeys.has(key)) {
                        extraParamKeys.add(key);
                    }
                });
            });
            
            const extraParamKeysArray = Array.from(extraParamKeys).sort();
            
            let dynamicHeaders = '';
            extraParamKeysArray.forEach(key => {
                const firstWord = key.split('_')[0];
                const displayName = firstWord.charAt(0).toUpperCase() + firstWord.slice(1);
                dynamicHeaders += '<th>' + displayName + '</th>';
            });
            
            // ✅ MODIFICACIÓN: Añadida columna Dir Mode
            document.querySelector('#strategies-table thead tr').innerHTML = 
                            '<th>#</th>' +
                            '<th>ID</th>' +
                            '<th>TF</th>' +
                            '<th>Symbols</th>' +
                            '<th>TP%</th>' +
                            '<th>SL%</th>' +
                            '<th>Amount</th>' +
                            '<th>Candles</th>' +
                            dynamicHeaders +
                            '<th>Status</th>';
            
            strategiesBody.innerHTML = sortedStrategies.map((strat, index) => {
                let statusBadge = '';
                if (strat.status === 'ACTIVE') {
                    statusBadge = '<span class="badge badge-active">Active</span>';
                } else if (strat.status === 'DEPRECATING') {
                    statusBadge = '<span class="badge badge-deprecating">Deprecating</span>';
                } else {
                    statusBadge = '<span class="badge badge-not-implemented">Not Impl.</span>';
                }
                
                const num = String(index + 1).padStart(2, '0');
                
                const fixedCols = 
                    '<td style="color: #8b949e; font-weight: 600;">' + num + '</td>' +
                    '<td>' + strat.id + '</td>' +
                    '<td>' + strat.timeframe + '</td>' +
                    '<td style="text-align: center; color: #58a6ff;">' + strat.symbols_count + '</td>';
                
                let commonCols = '';
                commonKeys.forEach(key => {
                    const value = strat[key] !== undefined ? strat[key] : 'N/A';
                    const display = key === 'order_amount' ? '$' + value : value;
                    commonCols += '<td>' + display + '</td>';
                });
                
                let extraCols = '';
                extraParamKeysArray.forEach(key => {
                    const value = strat[key];
                    const display = (value !== undefined && value !== 'N/A') ? value : '-';
                    extraCols += '<td>' + display + '</td>';
                });
                             
                return '<tr>' + fixedCols + commonCols + extraCols + '<td>' + statusBadge + '</td></tr>';
            }).join('');
        }
        
    } catch (error) {
        console.error('Error:', error);
    }
}

// =============================================================================
// GENERIC CHECKBOX FUNCTIONS (Used by Curves, Monthly, Correlation)
// =============================================================================

/**
 * Initialize strategy checkboxes for a tab
 * @param {string} containerId - ID of checkbox container div
 * @param {string} checkboxPrefix - Prefix for individual checkbox IDs (e.g., 'strat-')
 * @param {string} allCheckboxId - ID of "ALL STRATEGIES" checkbox
 * @returns {Promise<Array>} Array of strategies from API
 */
async function initStrategyCheckboxes(containerId, checkboxPrefix, allCheckboxId) {
    const res = await fetch('/api/bot-config');
    const data = await res.json();
    
    const strategies = data.strategies || [];
    
    const checkboxContainer = document.getElementById(containerId);
    const allCheckbox = checkboxContainer.querySelector(`#${allCheckboxId}`).parentElement;
    
    // Clear and re-add ALL checkbox
    checkboxContainer.innerHTML = '';
    checkboxContainer.appendChild(allCheckbox);
    allCheckbox.style.gridColumn = '1 / -1';
    
    // Create checkbox for each strategy
    strategies.forEach((strat) => {
        const div = document.createElement('div');
        div.className = 'checkbox-item';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = checkboxPrefix + strat.id;
        checkbox.value = strat.id;
        checkbox.checked = (strat.status === 'ACTIVE' || strat.status === 'DEPRECATING');
        
        const label = document.createElement('label');
        label.htmlFor = checkboxPrefix + strat.id;
        
        const displayName = strat.id.replace(/^\d{2}_/, '');
        label.textContent = '[' + strat.number + '] ' + displayName + ' (' + strat.status + ')';
        
        // Color coding by status
        if (strat.status === 'ACTIVE') {
            label.style.color = '#ffffff';
            label.style.fontWeight = '600';
        } else if (strat.status === 'DEPRECATING') {
            label.style.color = '#d29922';
        } else {
            label.style.color = '#8b949e';
            label.style.fontStyle = 'italic';
        }
        
        div.appendChild(checkbox);
        div.appendChild(label);
        checkboxContainer.appendChild(div);
    });
    
    // Setup "ALL" checkbox toggle - use cloneNode to avoid duplicate listeners
    const allCheckboxInput = document.getElementById(allCheckboxId);
    const newAllCheckbox = allCheckboxInput.cloneNode(true);
    allCheckboxInput.parentNode.replaceChild(newAllCheckbox, allCheckboxInput);
    
    newAllCheckbox.addEventListener('change', function() {
        const isChecked = this.checked;
        document.querySelectorAll(`#${containerId} input[type="checkbox"]`).forEach(cb => {
            if (cb.id !== allCheckboxId) cb.checked = isChecked;
        });
    });
    
    return strategies;
}

/**
 * Get selected strategy IDs from a checkbox container
 * @param {string} containerId - ID of checkbox container div
 * @returns {Array<string>} Array of selected strategy IDs (excludes "ALL")
 */
function getSelectedStrategies(containerId) {
    const selected = [];
    document.querySelectorAll(`#${containerId} input[type="checkbox"]:checked`).forEach(cb => {
        if (cb.value !== 'ALL') {
            selected.push(cb.value);
        }
    });
    return selected;
}

// =============================================================================
// END GENERIC CHECKBOX FUNCTIONS
// =============================================================================

async function loadEquityTab() {
    try {
        allStrategiesList = await initStrategyCheckboxes(
            'strategy-checkboxes',
            'strat-',
            'strat-all'
        );
        
        await updateEquityChart();
        
    } catch (error) {
        console.error('Error loading equity tab:', error);
    }
}
function extractNumberFromId(strategyId) {
    if (!strategyId || typeof strategyId !== 'string') {
        return '??';
    }
    
    const match = strategyId.match(/^(\d{2})_/);
    if (match) {
        return match[1];
    }
    
    const fallbackMatch = strategyId.match(/^(\d+)/);
    if (fallbackMatch) {
        return fallbackMatch[1].padStart(2, '0');
    }
    
    return '??';
}

async function updateEquityChart() {
    try {
        const selectedStrategies = getSelectedStrategies('strategy-checkboxes');
        
        if (selectedStrategies.length === 0) {
            alert('Please select at least one strategy');
            return;
        }
        
        const dateParams = getCurvesDateParams();
        const res = await fetch('/api/equity-data?strategies=' + selectedStrategies.join(',') + dateParams);
        const data = await res.json();
        
        if (!data.dates || data.dates.length === 0) {
            const strategiesStr = selectedStrategies.join(', ');
            alert(`No trades found for:\n${strategiesStr}\n\n${data.message || 'These strategies have no closed trades yet.'}`);
            return;
        }
        
        // Fetch BTC history with same date filters
        const btcRes = await fetch('/api/ref/history?timeframe=1Dutc' + dateParams);
        const btcData = await btcRes.json();
        
        document.getElementById('equity-metrics').style.display = 'block';
        document.getElementById('metric-num-trades').textContent = data.num_trades || 0;
        
        const profitPct = ((data.total_profit_usd / data.capital_assigned) * 100) || 0;
        document.getElementById('metric-profit-pct').textContent = (profitPct >= 0 ? '+' : '') + profitPct.toFixed(2) + '%';
        
        document.getElementById('metric-profit-usd').textContent = '$' + (data.total_profit_usd || 0);
        document.getElementById('metric-profit-factor').textContent = data.profit_factor || '-';
        document.getElementById('metric-weekly-win').textContent = (data.weekly_win_pct || 0) + '%';
        document.getElementById('metric-win-rate').textContent = Math.round(data.win_rate || 0) + '%';
        document.getElementById('metric-max-dd').textContent = (data.max_dd || 0) + '%';
        document.getElementById('metric-r-squared').textContent = data.r_squared || 0;
        document.getElementById('metric-sharpe').textContent = (data.sharpe_ratio || 0);
        
        const pctElement = document.getElementById('metric-profit-pct');
        applyMetricColor(pctElement, profitPct, 'positiveNegative');
        
        const puElement = document.getElementById('metric-profit-usd');
        applyMetricColor(puElement, data.total_profit_usd, 'positiveNegative');
        
        const pfElement = document.getElementById('metric-profit-factor');
        applyMetricColor(pfElement, data.profit_factor, 'profitFactor');
        
        const r2Element = document.getElementById('metric-r-squared');
        applyMetricColor(r2Element, data.r_squared, 'rSquared');
        
        const sharpeElement = document.getElementById('metric-sharpe');
        applyMetricColor(sharpeElement, data.sharpe_ratio, 'sharpe');
        
        if (equityChart) equityChart.destroy();
        if (drawdownChart) drawdownChart.destroy();
        
        const finalEquity = data.equity_pct[data.equity_pct.length - 1] || 0;
        const equityColor = finalEquity >= 0 ? COLORS.equityPositive : COLORS.equityNegative;
        
        // Prepare datasets
        const datasets = [{
            label: 'Equity (%)',
            data: data.equity_pct,
            borderColor: equityColor,
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1,
            yAxisID: 'y'
        }];
        
        // Add BTC dataset if data available
        // Add BTC dataset if data available (WITH DATE ALIGNMENT)
        if (btcData.success && btcData.dates && btcData.dates.length > 0) {
            // Normalize BTC dates: "YYYY-MM-DD HH:MM:SS" -> "YYYY-MM-DD"
            const btcDatesMap = {};
            btcData.dates.forEach((dateStr, idx) => {
                const normalizedDate = dateStr.split(' ')[0]; // Extract YYYY-MM-DD
                btcDatesMap[normalizedDate] = btcData.prices[idx];
            });
            
            // Align BTC prices with equity dates
            const alignedBtcPrices = data.dates.map(equityDate => {
                return btcDatesMap[equityDate] || null; // null if no match
            });
            
            // Only add if we have at least some overlap
            const validPrices = alignedBtcPrices.filter(p => p !== null);
            if (validPrices.length > 0) {
                datasets.push({
                    label: (btcData.symbol || 'BTC') + ' Price',
                    data: alignedBtcPrices,
                    borderColor: '#f59e0b',
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    tension: 0.1,
                    yAxisID: 'y2',
                    spanGaps: true  // Draw line across null values
                });
            }
        }
        const ctxEquity = document.getElementById('equityChart').getContext('2d');
        equityChart = new Chart(ctxEquity, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: { 
                        display: true,
                        position: 'top',
                        labels: {
                            color: COLORS.white,
                            font: { size: 14 }
                        }
                    },
                    title: { 
                        display: true, 
                        text: 'Equity Curve - ' + selectedStrategies.length + ' strategies selected',
                        color: CHART_DEFAULTS.titleColor,
                        font: { size: CHART_DEFAULTS.fontSize.title, weight: 'bold' }
                    }
                },
                scales: {
                    x: { 
                        ticks: { color: CHART_DEFAULTS.textColor, font: { size: CHART_DEFAULTS.fontSize.axis } }, 
                        grid: { 
                            color: CHART_DEFAULTS.gridColor,
                            drawBorder: true,
                            borderColor: CHART_DEFAULTS.borderColor,
                            borderWidth: CHART_DEFAULTS.borderWidth
                        } 
                    },
                    y: { 
                        type: 'linear',
                        display: true,
                        position: 'left',
                        ticks: { 
                            color: CHART_DEFAULTS.textColor,
                            font: { size: CHART_DEFAULTS.fontSize.axis },
                            callback: function(value) { return value.toFixed(1) + '%'; }
                        }, 
                        grid: { 
                            color: CHART_DEFAULTS.gridColor,
                            drawBorder: true,
                            borderColor: CHART_DEFAULTS.borderColor,
                            borderWidth: CHART_DEFAULTS.borderWidth
                        }
                    },
                    y2: {
                        type: 'linear',
                        display: datasets.length > 1,
                        position: 'right',
                        ticks: {
                            color: '#f59e0b',
                            font: { size: CHART_DEFAULTS.fontSize.axis },
                            callback: function(value) { return '$' + value.toLocaleString(); }
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            }
        });
        
        const ctxDD = document.getElementById('drawdownChart').getContext('2d');
        drawdownChart = new Chart(ctxDD, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [{
                    label: 'Drawdown (%)',
                    data: data.drawdown_pct,
                    borderColor: COLORS.drawdownRed,
                    backgroundColor: COLORS.drawdownRedAlpha,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1,
                    fill: true
                }]
            },
            options: getBaseChartConfig('Drawdown - ' + selectedStrategies.length + ' strategies selected', true)
        });
        
    } catch (error) {
        console.error('Error updating equity chart:', error);
        alert('Error loading equity data: ' + error.message);
    }
}

async function loadData() {
    if (isLoadingData) return;
    isLoadingData = true;
    try {
        const statusRes = await fetch('/api/status');
        if (!statusRes.ok) throw new Error('HTTP ' + statusRes.status);
        const status = await statusRes.json();
        
        requestAnimationFrame(() => {
            const totalPosEl = document.getElementById('total-positions');
            if (totalPosEl) totalPosEl.textContent = status.total_positions || 0;
            
            const totalProfit = status.total_profit || 0;
            const profitEl = document.getElementById('total-profit');
            if (profitEl) {
                profitEl.textContent = '$' + totalProfit.toFixed(0);
                profitEl.className = 'stat-value ' + (totalProfit >= 0 ? 'positive' : 'negative');
            }
            
            const openPnl = status.open_pnl || 0;
            const openPnlEl = document.getElementById('open-pnl');
            if (openPnlEl) {
                openPnlEl.textContent = '$' + openPnl.toFixed(0);
                openPnlEl.className = 'stat-value ' + (openPnl >= 0 ? 'positive' : 'negative');
            }
            
            const profitPct = status.profit_pct || 0;
            const profitPctEl = document.getElementById('profit-pct');
            if (profitPctEl) {
                profitPctEl.textContent = (profitPct >= 0 ? '+' : '') + profitPct.toFixed(2) + '%';
                profitPctEl.className = 'stat-value ' + (profitPct >= 0 ? 'positive' : 'negative');
            }
            
            const tradesNumEl = document.getElementById('trades-num');
            if (tradesNumEl) tradesNumEl.textContent = status.num_trades || 0;
            
            const tradesPct = status.trades_pct || 0;
            const tradesPctEl = document.getElementById('trades-pct');
            if (tradesPctEl) tradesPctEl.textContent = tradesPct.toFixed(1) + '%';
            
            const refPrice = status.ref_price || 0;
            const refPriceEl = document.getElementById('btc-price');
            if (refPriceEl) refPriceEl.textContent = '$' + refPrice.toLocaleString('es-ES', {minimumFractionDigits: 0, maximumFractionDigits: 0});

            if (status.ref_symbol) {
                const refLabel = document.getElementById('ref-symbol-label');
                if (refLabel) refLabel.textContent = status.ref_symbol.replace('USDT', '') + ' Price';
            }

        });
        
        // Load exposure data with dynamic limits from backend
        try {
            const exposureRes = await fetch('/api/risk/exposure');
            if (exposureRes.ok) {
                const exposureData = await exposureRes.json();
                if (exposureData.success) {

                    const grossPct = exposureData.metrics.gross_exposure_pct;
                    const netPct = exposureData.metrics.net_exposure_pct;
                    
                    // Update header cards (always blue)
                    const grossEl = document.getElementById('exposure-gross');
                    const netEl = document.getElementById('exposure-net');
                    
                    if (grossEl) {
                        grossEl.textContent = grossPct.toFixed(1) + '%';
                        grossEl.style.color = '#58a6ff';  // Blue
                    }
                    
                    if (netEl) {
                        netEl.textContent = (netPct >= 0 ? '+' : '') + netPct.toFixed(1) + '%';
                        netEl.style.color = '#58a6ff';  // Blue
                    }
                }
            }
        } catch (error) {
            console.error('Error loading exposure:', error);
        }
        
        const posRes = await fetch('/api/positions');
        if (!posRes.ok) throw new Error('HTTP ' + posRes.status);
        const positions = await posRes.json();
        
        requestAnimationFrame(() => {
            if (currentTab === 'positions') renderPositions(positions);
        });
        
        const tradesRes = await fetch('/api/trades/recent');
        if (!tradesRes.ok) throw new Error('HTTP ' + tradesRes.status);
        const trades = await tradesRes.json();

        
        requestAnimationFrame(() => {
            const tradesBody = document.getElementById('trades-body');
            if (!trades || trades.length === 0) {
                tradesBody.innerHTML = '<tr><td colspan="10" style="text-align: center;">No trades</td></tr>';
            } else {
                tradesBody.innerHTML = trades.map(trade => {
                    const profitClass = trade.PROFIT >= 0 ? 'direction-long' : 'direction-short';
                    let reasonBadge = '';
                    if (trade.REASON_OUT === 'TP') {
                        reasonBadge = '<span class="badge badge-tp">TP</span>';
                    } else if (trade.REASON_OUT === 'SL') {
                        reasonBadge = '<span class="badge badge-sl">SL</span>';
                    } else {
                        reasonBadge = '<span class="badge badge-timeout">TIMEOUT</span>';
                    }
                    return '<tr>' +
                        '<td>' + trade.CLOSE_AT + '</td>' +
                        '<td>' + trade.STRATEGY + '</td>' +
                        '<td>' + trade.SYMBOL + '</td>' +
                        '<td class="direction-' + trade.DIRECTION.toLowerCase() + '">' + trade.DIRECTION + '</td>' +
                        '<td>$' + parseFloat(trade.USDT_AMOUNT || 0).toFixed(2) + '</td>' +
                        '<td>$' + parseFloat(trade.PRICE_ENTRY || 0).toFixed(2) + '</td>' +
                        '<td>$' + parseFloat(trade.PRICE_CLOSE || 0).toFixed(2) + '</td>' +
                        '<td class="' + profitClass + '">' + (trade.PROFIT >= 0 ? '+' : '') + '$' + trade.PROFIT.toFixed(2) + '</td>' +
                        '<td class="' + profitClass + '">' + (trade.PROFIT_PCT >= 0 ? '+' : '') + trade.PROFIT_PCT.toFixed(2) + '%</td>' +
                        '<td>' + reasonBadge + '</td>' +
                        '</tr>';
                }).join('');
            }
        });
        
    } catch (error) {
        console.error('Error:', error);
    } finally {
        isLoadingData = false;
    }
}

function toggleAutoScroll() {
    autoScroll = !autoScroll;
    document.getElementById('auto-scroll-btn').textContent = autoScroll ? 'Auto' : 'Manual';
}

function clearLogs() {
    if (confirm('Clear all logs?')) {
        document.getElementById('logs-content').innerHTML = '<div class="log-line default">Logs cleared</div>';
    }
}

const POLLING_CONFIG = {
    data: {
        active: 5000,
        inactive: 90000
    },
    logs: {
        active: 2000,
        inactive: 90000
    }
};

let dataInterval, logsInterval;

function updateClock() {
    const now = new Date();
    const hours = String(now.getUTCHours()).padStart(2, '0');
    const minutes = String(now.getUTCMinutes()).padStart(2, '0');
    const seconds = String(now.getUTCSeconds()).padStart(2, '0');
    const clockEl = document.getElementById('live-clock');
    if (clockEl) {
        clockEl.textContent = `${hours}:${minutes}:${seconds}`;
    }
}

function startPolling() {
    stopPolling();
    
    loadData();
    loadLogs();
    updateClock();
    
    const isActive = !document.hidden;
    const dataDelay = isActive ? POLLING_CONFIG.data.active : POLLING_CONFIG.data.inactive;
    const logsDelay = isActive ? POLLING_CONFIG.logs.active : POLLING_CONFIG.logs.inactive;
    
    dataInterval = setInterval(() => loadData().catch(console.error), dataDelay);
    logsInterval = setInterval(() => loadLogs().catch(console.error), logsDelay);
    setInterval(updateClock, 1000);
}

function stopPolling() {
    if (dataInterval) clearInterval(dataInterval);
    if (logsInterval) clearInterval(logsInterval);
}

document.addEventListener('visibilitychange', () => {
    if (document.hidden) stopPolling();
    else startPolling();
});

const savedView = 'compact';
currentPositionsView = savedView;

async function initializeDashboard() {
    document.querySelectorAll('.view-btn').forEach(btn => {
        if (btn.textContent.toLowerCase().includes(savedView)) btn.classList.add('active');
        else btn.classList.remove('active');
    });
    
    const backendReady = await waitForBackend();
    await loadQualityThresholds();
    await loadBotConfig();
    startPolling();
}


// =============================================================================
// RISK CONTROL TAB
// =============================================================================

async function loadRiskTab() {
    try {
        const exposureRes = await fetch('/api/risk/exposure');
        const exposureData = await exposureRes.json();
        
        if (!exposureData.success) {
            console.error('Failed to load risk exposure:', exposureData.error);
            return;
        }
        
        const metrics   = exposureData.metrics;
        const strategies = exposureData.strategies;

        if (exposureData.limits) {
            const configGrossEl = document.getElementById('risk-config-max-gross');
            const configNetEl   = document.getElementById('risk-config-max-net');
            if (configGrossEl) configGrossEl.textContent = exposureData.limits.max_gross.toFixed(1) + '%';
            if (configNetEl)   configNetEl.textContent   = exposureData.limits.max_net.toFixed(1) + '%';
        }

        updateRiskCards(metrics, strategies);
        
    } catch (error) {
        console.error('Error loading risk tab:', error);
    }
}

// =============================================================================
// QUALITY CONTROL TAB
// =============================================================================
async function loadQualityTab() {
    try {
        // Load execution quality + target deviation in a single request

        // Load execution quality + target deviation in a single request
        const qualityRes  = await fetch('/api/quality/all');
        const qualityData = await qualityRes.json();

        if (qualityData.success) {
            renderExecutionTable(qualityData.data.execution);
            renderTargetDeviationTable(qualityData.data.deviation);
        } else {
            const errorHtml =
                '<div style="text-align: center; color: #f85149; padding: 40px;">' +
                (qualityData.error || 'Error loading quality data') +
                '</div>';
            document.getElementById('execution-table-container').innerHTML  = errorHtml;
            document.getElementById('deviation-table-container').innerHTML  = errorHtml;
        }

        // Initialize win rate evolution checkboxes
        await initStrategyCheckboxes(
            'winrate-strategy-checkboxes',
            'winrate-strat-',
            'winrate-strat-all'
        );

    } catch (error) {
        console.error('Error loading quality tab:', error);
        const errorHtml = '<div style="text-align: center; color: #f85149; padding: 40px;">Error loading data</div>';
        document.getElementById('execution-table-container').innerHTML = errorHtml;
        document.getElementById('deviation-table-container').innerHTML = errorHtml;
    }
}

function renderExecutionTable(data) {
    const container = document.getElementById('execution-table-container');
    
    if (!data || Object.keys(data).length === 0) {
        container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No data available</div>';
        return;
    }
    
    // Sort by strategy ID
    const sortedStrategies = Object.keys(data).sort();
    
    let html = '<table><thead><tr>' +
        '<th>#</th>' +
        '<th>Strategy</th>' +
        '<th>Close Slip</th>' +
        '<th>Status</th>' +
        '<th>TP Slip</th>' +
        '<th>Status</th>' +
        '<th>SL Slip</th>' +
        '<th>Status</th>' +
        '<th>Avg Latency</th>' +
        '<th>Status</th>' +
        '<th>Total Trades</th>' +
        '</tr></thead><tbody>';
    
    sortedStrategies.forEach((strategyId, index) => {
        const strat = data[strategyId];
        const num = String(index + 1).padStart(2, '0');
        
        // Helper function for status color
        function getStatusColor(status) {
            if (status === 'HEALTHY') return COLORS.healthy;
            if (status === 'WARNING') return COLORS.warning;
            if (status === 'DANGER') return COLORS.danger;
            return '#8b949e';
        }
        
        // Format slippage values
        function formatSlippage(value) {
            if (value === null || value === undefined) return '-';
            return (value >= 0 ? '+' : '') + value.toFixed(2) + '%';
        }
        
        const closeSlippageText = formatSlippage(strat.avg_close_slippage_pct);
        const tpSlippageText = formatSlippage(strat.avg_tp_slippage_pct);
        const slSlippageText = formatSlippage(strat.avg_sl_slippage_pct);
        
        const latencyText = strat.avg_latency_sec !== null ? 
            strat.avg_latency_sec.toFixed(3) + 's' : '-';
        
        html += '<tr>' +
            '<td style="color: #8b949e; font-weight: 600;">' + num + '</td>' +
            '<td>' + strategyId + '</td>' +
            '<td>' + closeSlippageText + '</td>' +
            '<td style="color: ' + getStatusColor(strat.close_slippage_status) + '; font-weight: 700; text-transform: uppercase;">' + strat.close_slippage_status + '</td>' +
            '<td>' + tpSlippageText + '</td>' +
            '<td style="color: ' + getStatusColor(strat.tp_slippage_status) + '; font-weight: 700; text-transform: uppercase;">' + strat.tp_slippage_status + '</td>' +
            '<td>' + slSlippageText + '</td>' +
            '<td style="color: ' + getStatusColor(strat.sl_slippage_status) + '; font-weight: 700; text-transform: uppercase;">' + strat.sl_slippage_status + '</td>' +
            '<td>' + latencyText + '</td>' +
            '<td style="color: ' + getStatusColor(strat.latency_status) + '; font-weight: 700; text-transform: uppercase;">' + strat.latency_status + '</td>' +
            '<td>' + strat.total_trades + '</td>' +
            '</tr>';
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
}
function renderTargetDeviationTable(data) {
    const container = document.getElementById('deviation-table-container');
    
    if (!data || Object.keys(data).length === 0) {
        container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No data available</div>';
        return;
    }
    
    // Sort by strategy ID
    const sortedStrategies = Object.keys(data).sort();
    
    let html = '<table><thead><tr>' +
        '<th>#</th>' +
        '<th>Strategy</th>' +
        '<th>TP Real%</th>' +
        '<th>TP Target%</th>' +
        '<th>TP Dev</th>' +
        '<th>SL Real%</th>' +
        '<th>SL Target%</th>' +
        '<th>SL Dev</th>' +
        '<th>Total Trades</th>' +
        '</tr></thead><tbody>';
    
    sortedStrategies.forEach((strategyId, index) => {
        const strat = data[strategyId];
        const num = String(index + 1).padStart(2, '0');
        
        // Helper function for deviation color
        function getDeviationColor(deviation) {
            if (deviation === null || deviation === undefined) return '#8b949e';
            
            // Positive deviation is ALWAYS good (we got more profit than expected)
            if (deviation > 0) return '#3fb950';  // 🟢 Green
            
            // Negative deviation: apply thresholds
            const absDev = Math.abs(deviation);
            if (absDev < 0.2) return '#3fb950';  
            if (absDev < 0.5) return COLORS.warning;  
            return COLORS.danger;                   
        }
        
        // Format percentage values
        function formatPct(value) {
            if (value === null || value === undefined) return '-';
            return (value >= 0 ? '+' : '') + value.toFixed(2) + '%';
        }
        
        const tpRealText = formatPct(strat.tp_real_pct);
        const tpTargetText = formatPct(strat.tp_target_pct);
        const tpDevText = formatPct(strat.tp_deviation);
        
        const slRealText = formatPct(strat.sl_real_pct);
        const slTargetText = formatPct(strat.sl_target_pct);
        const slDevText = formatPct(strat.sl_deviation);
        
        const tpDevColor = getDeviationColor(strat.tp_deviation);
        const slDevColor = getDeviationColor(strat.sl_deviation);
        
        // Total trades = TP trades + SL trades
        const totalTrades = strat.tp_trades + strat.sl_trades + (strat.timeout_trades || 0);
        
        html += '<tr>' +
            '<td style="color: #8b949e; font-weight: 600;">' + num + '</td>' +
            '<td>' + strategyId + '</td>' +
            '<td>' + tpRealText + '</td>' +
            '<td>' + tpTargetText + '</td>' +
            '<td style="color: ' + tpDevColor + '; font-weight: 600;">' + tpDevText + '</td>' +
            '<td>' + slRealText + '</td>' +
            '<td>' + slTargetText + '</td>' +
            '<td style="color: ' + slDevColor + '; font-weight: 600;">' + slDevText + '</td>' +
            '<td>' + totalTrades + '</td>' +
            '</tr>';
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
}
// =============================================================================
// END QUALITY CONTROL TAB
// =============================================================================

async function updateRiskCards(metrics, strategies) {
    const grossPct  = metrics.gross_exposure_pct;
    const netPct    = metrics.net_exposure_pct;
    const longPct   = metrics.long_exposure_pct;
    const shortPct  = metrics.short_exposure_pct;

    const longUsdt  = metrics.long_usdt  || 0;
    const shortUsdt = metrics.short_usdt || 0;

    const grossEl = document.getElementById('risk-gross-exp');
    if (grossEl) {
        grossEl.textContent = grossPct.toFixed(1) + '%';
        grossEl.style.color = '#58a6ff';
    }

    const netUsdt = longUsdt - shortUsdt;
    const netEl = document.getElementById('risk-net-exp');
    if (netEl) {
        netEl.textContent = (netPct >= 0 ? '+' : '') + netPct.toFixed(1) + '% | ' + (netUsdt >= 0 ? '+$' : '-$') + Math.abs(netUsdt).toFixed(0);
        netEl.style.color = '#58a6ff';
    }

    const longEl = document.getElementById('risk-long-exp');
    if (longEl) longEl.textContent = longPct.toFixed(1) + '% | $' + longUsdt.toFixed(0);

    const shortEl = document.getElementById('risk-short-exp');
    if (shortEl) shortEl.textContent = shortPct.toFixed(1) + '% | $' + shortUsdt.toFixed(0);
}

// =============================================================================
// WIN RATE EVOLUTION CHART (Quality Control)
// =============================================================================

let winRateChart = null;

function clearWinRateDates() {
    document.getElementById('winrate-date-from').value = '';
    document.getElementById('winrate-date-to').value = '';
}

function getWinRateDateParams() {
    const dateFrom = document.getElementById('winrate-date-from').value;
    const dateTo = document.getElementById('winrate-date-to').value;
    let params = '';
    if (dateFrom) params += '&date_from=' + dateFrom;
    if (dateTo) params += '&date_to=' + dateTo;
    return params;
}

async function updateWinRateChart() {
    try {
        const selectedStrategies = getSelectedStrategies('winrate-strategy-checkboxes');
        
        if (selectedStrategies.length === 0) {
            alert('Please select at least one strategy');
            return;
        }
        
        const dateParams = getWinRateDateParams();
        const res = await fetch('/api/quality/winrate-evolution?strategies=' + selectedStrategies.join(',') + dateParams);
        const data = await res.json();
        
        if (!data.success) {
            alert('Error loading win rate data: ' + (data.error || 'Unknown error'));
            return;
        }
        
        if (!data.dates || data.dates.length === 0) {
            alert('No trades found for selected strategies in this date range');
            return;
        }
        
        // Destroy existing chart
        if (winRateChart) {
            winRateChart.destroy();
            winRateChart = null;
        }
        
        // Create chart
        const ctx = document.getElementById('winRateChart').getContext('2d');
        winRateChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [{
                    label: 'Cumulative Win Rate (%)',
                    data: data.winrate,
                    borderColor: '#58a6ff',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 2,
                    pointBackgroundColor: '#58a6ff',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#ffffff',
                            font: { size: 14 }
                        }
                    },
                    title: {
                        display: true,
                        text: 'Win Rate Evolution - ' + selectedStrategies.length + ' strategies (' + data.total_trades + ' trades)',
                        color: CHART_DEFAULTS.titleColor,
                        font: { size: CHART_DEFAULTS.fontSize.title, weight: 'bold' }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: CHART_DEFAULTS.textColor,
                            font: { size: CHART_DEFAULTS.fontSize.axis }
                        },
                        grid: {
                            color: CHART_DEFAULTS.gridColor,
                            drawBorder: true,
                            borderColor: CHART_DEFAULTS.borderColor,
                            borderWidth: CHART_DEFAULTS.borderWidth
                        }
                    },
                    y: {
                        ticks: {
                            color: CHART_DEFAULTS.textColor,
                            font: { size: CHART_DEFAULTS.fontSize.axis },
                            callback: function(value) { return value.toFixed(1) + '%'; }
                        },
                        grid: {
                            color: CHART_DEFAULTS.gridColor,
                            drawBorder: true,
                            borderColor: CHART_DEFAULTS.borderColor,
                            borderWidth: CHART_DEFAULTS.borderWidth
                        }
                    }
                }
            }
        });
        
    } catch (error) {
        console.error('Error updating win rate chart:', error);
        alert('Error loading win rate chart: ' + error.message);
    }
}
// =============================================================================
// PERIOD ANALYSIS — Monthly & Weekly subtabs
// =============================================================================

let monthlyPeriodChart = null;
let weeklyPeriodChart  = null;

// --- Inner tab switching ---

function switchPeriodInnerTab(tab) {
    document.getElementById('period-inner-monthly').style.display = tab === 'monthly' ? '' : 'none';
    document.getElementById('period-inner-weekly').style.display  = tab === 'weekly'  ? '' : 'none';

    document.getElementById('period-inner-btn-monthly').classList.toggle('active', tab === 'monthly');
    document.getElementById('period-inner-btn-weekly').classList.toggle('active',  tab === 'weekly');
}

// --- Date helpers ---

function clearMonthlyDates() {
    document.getElementById('monthly-date-from').value = '';
    document.getElementById('monthly-date-to').value   = '';
}

function clearWeeklyDates() {
    document.getElementById('weekly-date-from').value = '';
    document.getElementById('weekly-date-to').value   = '';
}

function getMonthlyDateParams() {
    const from = document.getElementById('monthly-date-from').value;
    const to   = document.getElementById('monthly-date-to').value;
    let params = '';
    if (from) params += '&date_from=' + from;
    if (to)   params += '&date_to='   + to;
    return params;
}

function getWeeklyDateParams() {
    const from = document.getElementById('weekly-date-from').value;
    const to   = document.getElementById('weekly-date-to').value;
    let params = '';
    if (from) params += '&date_from=' + from;
    if (to)   params += '&date_to='   + to;
    return params;
}

// --- Chart builder (shared) ---

function buildPeriodChart(canvasId, labels, profits, winRates, existingChart) {
    if (existingChart) {
        existingChart.destroy();
        existingChart = null;
    }

    const profitColors = profits.map(v => v >= 0 ? COLORS.green : COLORS.red);
    const ctx = document.getElementById(canvasId).getContext('2d');

    return new Chart(ctx, {
        data: {
            labels,
            datasets: [
                {
                    type:            'line',
                    label:           'Profit %',
                    data:            profits,
                    backgroundColor: 'transparent',
                    borderColor:     COLORS.green,
                    pointBackgroundColor: profits.map(v => v >= 0 ? COLORS.green : COLORS.red),
                    pointRadius:     4,
                    tension:         0.1,
                    borderWidth:     2,
                    yAxisID:         'yProfit',
                    order:           2
                },
                {
                    type:                 'line',
                    label:                'Win Rate %',
                    data:                 winRates,
                    borderColor:          COLORS.blue,
                    backgroundColor:      'transparent',
                    borderWidth:          2,
                    pointRadius:          4,
                    pointBackgroundColor: COLORS.blue,
                    tension:              0.1,
                    yAxisID:              'yWinRate',
                    order:                1
                }
            ]
        },
        options: {
            responsive:          true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    display:  true,
                    position: 'top',
                    labels: { color: COLORS.white, font: { size: 13 } }
                },
                title: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%`
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: COLORS.white, font: { size: 14 }, maxRotation: 45 },
                    grid:  { color: CHART_DEFAULTS.gridColor, borderColor: CHART_DEFAULTS.borderColor, borderWidth: CHART_DEFAULTS.borderWidth }
                },
                yProfit: {
                    type:     'linear',
                    position: 'left',
                    title:    { display: true, text: 'Profit %', color: COLORS.green, font: { size: 13 } },
                    ticks:    { color: COLORS.white, font: { size: 14 }, callback: v => v.toFixed(1) + '%' },
                    grid:     { color: CHART_DEFAULTS.gridColor, borderColor: CHART_DEFAULTS.borderColor, borderWidth: CHART_DEFAULTS.borderWidth }
                },
                yWinRate: {
                    type:     'linear',
                    position: 'right',
                    min:      0,
                    max:      100,
                    title:    { display: true, text: 'Win Rate %', color: COLORS.blue, font: { size: 13 } },
                    ticks:    { color: COLORS.white, font: { size: 14 }, callback: v => v.toFixed(0) + '%' },
                    grid:     { drawOnChartArea: false }
                }
            }
        }
    });
}

// --- Cards builder (shared) ---

function buildPeriodCards(data, containerId, labelKey) {
    const container = document.getElementById(containerId);

    if (!data || data.length === 0) {
        container.innerHTML = '<div style="text-align:center; color:#8b949e; padding:40px;">No data for selected period</div>';
        return;
    }

    const html = '<div style="display:flex; flex-wrap:wrap; gap:12px; padding:10px 0;">' +
        data.map(row => {
            const profitColor  = row.profit_pct >= 0 ? COLORS.green : COLORS.red;
            const profitUsdColor = row.profit_usd >= 0 ? COLORS.green : COLORS.red;
            const prefixPct    = row.profit_pct >= 0 ? '+' : '';
            const prefixUsd    = row.profit_usd >= 0 ? '+$' : '$';
            const label        = row[labelKey] || row.week_label || row.month_name;

            return `<div style="background:#1c2128; border:1px solid #21262d; border-radius:8px; padding:15px 20px; min-width:120px; text-align:center;">
                <div style="color:${profitColor}; font-size:18px; font-weight:700; margin-bottom:2px;">${prefixPct}${row.profit_pct.toFixed(1)}%</div>
                <div style="color:${profitUsdColor}; font-size:13px; font-weight:600; margin-bottom:6px;">${prefixUsd}${row.profit_usd.toFixed(0)}</div>
                <div style="color:${COLORS.blue}; font-size:13px; font-weight:600; margin-bottom:6px;">WR: ${row.win_rate.toFixed(1)}%</div>
                <div style="color:#8b949e; font-size:11px; margin-bottom:6px;">${row.num_trades} trades</div>
                <div style="color:#58a6ff; font-size:12px; font-weight:600; text-transform:uppercase;">${label}</div>
            </div>`;
        }).join('') +
        '</div>';

    container.innerHTML = html;
}

// --- Monthly ---

async function loadMonthlyAnalysis() {
    try {
        const selectedStrategies = getSelectedStrategies('monthly-strategy-checkboxes');

        if (selectedStrategies.length === 0) {
            alert('Please select at least one strategy');
            return;
        }

        const dateParams = getMonthlyDateParams();
        const res  = await fetch('/api/monthly-analysis?strategies=' + selectedStrategies.join(',') + dateParams);
        const data = await res.json();

        if (!data || data.length === 0) {
            document.getElementById('monthly-container').innerHTML =
                '<div style="text-align:center; color:#8b949e; padding:40px;">No data for selected period</div>';
            if (monthlyPeriodChart) { monthlyPeriodChart.destroy(); monthlyPeriodChart = null; }
            return;
        }

        monthlyPeriodChart = buildPeriodChart(
            'monthlyPeriodChart',
            data.map(d => d.month_name),
            data.map(d => d.profit_pct),
            data.map(d => d.win_rate),
            monthlyPeriodChart
        );

        buildPeriodCards(data, 'monthly-container', 'month_name');

    } catch (error) {
        console.error('Error loading monthly analysis:', error);
    }
}

// --- Weekly ---

async function loadWeeklyAnalysis() {
    try {
        const selectedStrategies = getSelectedStrategies('weekly-strategy-checkboxes');

        if (selectedStrategies.length === 0) {
            alert('Please select at least one strategy');
            return;
        }

        const dateParams = getWeeklyDateParams();
        const res  = await fetch('/api/weekly-analysis?strategies=' + selectedStrategies.join(',') + dateParams);
        const data = await res.json();

        if (!data || data.length === 0) {
            document.getElementById('weekly-container').innerHTML =
                '<div style="text-align:center; color:#8b949e; padding:40px;">No data for selected period</div>';
            if (weeklyPeriodChart) { weeklyPeriodChart.destroy(); weeklyPeriodChart = null; }
            return;
        }

        weeklyPeriodChart = buildPeriodChart(
            'weeklyPeriodChart',
            data.map(d => d.week_label),
            data.map(d => d.profit_pct),
            data.map(d => d.win_rate),
            weeklyPeriodChart
        );

        buildPeriodCards(data, 'weekly-container', 'week_label');

    } catch (error) {
        console.error('Error loading weekly analysis:', error);
    }
}

// --- Init Period tab ---

async function initPeriodTab() {
    try {
        await initStrategyCheckboxes('monthly-strategy-checkboxes', 'monthly-strat-', 'monthly-strat-all');
        await initStrategyCheckboxes('weekly-strategy-checkboxes',  'weekly-strat-',  'weekly-strat-all');
    } catch (error) {
        console.error('Error initializing period tab:', error);
    }
}

// =============================================================================
// END PERIOD ANALYSIS
// =============================================================================


// =============================================================================
// END WIN RATE EVOLUTION CHART
// =============================================================================
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeDashboard);
} else {
    initializeDashboard();
}

window.addEventListener('pageshow', function(event) {
    if (event.persisted) {
        initializeDashboard();
    }
});