# =============================================================================
# PSG Analysis Tool — Draggable Tiles + Rich Status + Progress + XGBoost
# =============================================================================

# Use Shiny's validate/need explicitly (avoid jsonlite::validate)
validate <- shiny::validate
need     <- shiny::need

suppressPackageStartupMessages({
  library(shiny)
  library(shinyjqui)
  library(jsonlite)
  library(ggplot2)
  library(plotly)
  library(DT)
  library(htmltools)
})

# Disable notifications globally (we use the Status tile instead)
showNotification <- function(...) invisible(NULL)

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0) y else x

# ---- Colors & constants ----
page_bg      <- "#F2F0E9"
nb_bg        <- "#FAF8F1"
menu_border  <- "#E0E0E0"
brand_color  <- "#751F2C"

red_pal <- list(
  base = "#751F2C",
  light = "#A04559",
  dark  = "#55131E",
  mid   = "#8A2C3C",
  pale  = "#C46C7B"
)

TILE_H <- 320L

DEFAULT_STATE <- list(
  order = c(
    "tile_controls",
    "tile_status",
    "tile_index",
    "tile_summary",
    "tile_roc",
    "tile_hist",
    "tile_imp",
    "tile_preds"
  )
)

# ---- Icons (Lucide-like inline SVG) ----
MOVE_SVG <- HTML('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
  viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
  stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-move">
  <path d="M12 2v20"/><path d="m15 19-3 3-3-3"/><path d="m19 9 3 3-3 3"/>
  <path d="M2 12h20"/><path d="m5 9-3 3 3 3"/><path d="m9 5 3-3 3 3"/>
</svg>')

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
has_pkg <- function(p) requireNamespace(p, quietly = TRUE)

timestamp_est <- function(fmt = "%Y-%m-%d %H:%M:%S") {
  format(Sys.time(), fmt, tz = "America/Toronto")
}

extract_entity <- function(path, entity = "sub") {
  pat <- paste0(entity, "-[A-Za-z0-9]+")
  m <- regmatches(path, regexpr(pat, path))
  if (length(m) == 0 || is.na(m) || !nzchar(m)) return(NA_character_)
  m
}

safe_read_tsv <- function(path) {
  tryCatch(
    read.delim(path, sep = "\t", header = TRUE, stringsAsFactors = FALSE, check.names = FALSE),
    error = function(e) NULL
  )
}

detect_bids_root <- function(unzipped_dir) {
  if (file.exists(file.path(unzipped_dir, "participants.tsv"))) return(unzipped_dir)
  kids <- list.dirs(unzipped_dir, recursive = FALSE, full.names = TRUE)
  kids <- kids[basename(kids) != "__MACOSX"]
  if (length(kids) == 1 && file.exists(file.path(kids[1], "participants.tsv"))) return(kids[1])
  pts <- list.files(unzipped_dir, pattern = "^participants\\.tsv$", recursive = TRUE, full.names = TRUE)
  if (length(pts)) return(dirname(pts[1]))
  unzipped_dir
}

index_bids_edf <- function(root) {
  edfs <- list.files(root, pattern = "\\.edf$", recursive = TRUE, full.names = TRUE)
  if (!length(edfs)) return(NULL)
  
  # Prefer BIDS EEG convention: inside /eeg/ and ends with _eeg.edf
  prefer <- grepl("[/\\\\]eeg[/\\\\]", edfs) & grepl("_eeg\\.edf$", edfs, ignore.case = TRUE)
  if (any(prefer)) edfs <- edfs[prefer]
  
  data.frame(
    participant_id = vapply(edfs, extract_entity, character(1), entity = "sub"),
    session_id     = vapply(edfs, extract_entity, character(1), entity = "ses"),
    edf_path       = edfs,
    filename       = basename(edfs),
    stringsAsFactors = FALSE
  )
}

# Your dataset naming:
# EDF: ..._eeg.edf
# channels: ..._channels.tsv (without the _eeg)
guess_channels_tsv <- function(edf_path) {
  dir  <- dirname(edf_path)
  base <- sub("\\.edf$", "", basename(edf_path), ignore.case = TRUE)
  
  # candidate A: base + _channels.tsv (sometimes exists)
  candA <- file.path(dir, paste0(base, "_channels.tsv"))
  
  # candidate B: strip trailing _eeg then + _channels.tsv  (THIS matches your screenshot)
  base2 <- sub("_eeg$", "", base, ignore.case = TRUE)
  candB <- file.path(dir, paste0(base2, "_channels.tsv"))
  
  # candidate C: plain channels.tsv in folder
  candC <- file.path(dir, "channels.tsv")
  
  cand <- c(candA, candB, candC)
  cand <- cand[file.exists(cand)]
  cand[1] %||% NA_character_
}

choose_eeg_channels <- function(edf_path, max_n = 2) {
  if (!has_pkg("edfReader")) return(character(0))
  
  ch_tsv <- guess_channels_tsv(edf_path)
  if (is.character(ch_tsv) && file.exists(ch_tsv)) {
    ch <- safe_read_tsv(ch_tsv)
    if (!is.null(ch) && nrow(ch)) {
      nn <- tolower(names(ch)); names(ch) <- nn
      if (!("name" %in% names(ch))) return(character(0))
      if (!("type" %in% names(ch))) ch$type <- ""
      if (!("status" %in% names(ch))) ch$status <- "good"
      
      ok <- toupper(ch$type) == "EEG" & (tolower(ch$status) %in% c("good", "") | is.na(ch$status))
      picks <- ch$name[ok]
      picks <- picks[!is.na(picks) & nzchar(picks)]
      if (length(picks)) return(head(picks, max_n))
    }
  }
  
  # fallback: first non-annotation signals
  hdr <- edfReader::readEdfHeader(edf_path)
  sigs <- edfReader::readEdfSignals(hdr, from = 0, till = 1, simplify = FALSE)
  ord <- names(sigs)[!vapply(sigs, function(s) isTRUE(s$isAnnotation), logical(1))]
  ord <- ord[!is.na(ord) & nzchar(ord)]
  head(ord, max_n)
}

read_edf_window <- function(edf_path, channel_labels, from_s, till_s) {
  hdr <- edfReader::readEdfHeader(edf_path)
  sigs <- edfReader::readEdfSignals(hdr, signals = channel_labels, from = from_s, till = till_s, simplify = FALSE)
  
  out <- list()
  for (nm in names(sigs)) {
    s <- sigs[[nm]]
    if (isTRUE(s$isAnnotation)) next
    x <- suppressWarnings(as.numeric(s$signal))
    fs <- suppressWarnings(as.numeric(s$sRate))
    out[[nm]] <- list(x = x, fs = fs)
  }
  out
}

# Minimal features (fast + stable)
skewness1 <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 3) return(NA_real_)
  m <- mean(x); s <- sd(x)
  if (!is.finite(s) || s == 0) return(NA_real_)
  mean(((x - m) / s)^3)
}
kurtosis1 <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 4) return(NA_real_)
  m <- mean(x); s <- sd(x)
  if (!is.finite(s) || s == 0) return(NA_real_)
  mean(((x - m) / s)^4) - 3
}
rms1 <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  sqrt(mean(x^2))
}
line_length <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 2) return(NA_real_)
  sum(abs(diff(x)))
}
zero_cross_rate <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 2) return(NA_real_)
  s <- sign(x)
  mean(abs(diff(s)) == 2)
}

band_powers <- function(x, fs, bands = list(
  delta = c(0.5, 4),
  theta = c(4, 8),
  alpha = c(8, 13),
  beta  = c(13, 30),
  gamma = c(30, 45)
)) {
  x <- x[is.finite(x)]
  if (length(x) < 256 || !is.finite(fs) || fs <= 0) {
    out <- list(total_0p5_45 = NA_real_, sef95_0p5_45 = NA_real_)
    for (nm in names(bands)) {
      out[[paste0("bp_", nm)]] <- NA_real_
      out[[paste0("rp_", nm)]] <- NA_real_
    }
    return(out)
  }
  
  sp <- tryCatch(stats::spectrum(x, plot = FALSE, fast = TRUE, detrend = TRUE, taper = 0.1, frequency = fs),
                 error = function(e) NULL)
  if (is.null(sp)) {
    out <- list(total_0p5_45 = NA_real_, sef95_0p5_45 = NA_real_)
    for (nm in names(bands)) {
      out[[paste0("bp_", nm)]] <- NA_real_
      out[[paste0("rp_", nm)]] <- NA_real_
    }
    return(out)
  }
  
  f <- sp$freq
  p <- sp$spec
  keep <- which(f >= 0.5 & f <= 45)
  f2 <- f[keep]; p2 <- p[keep]
  
  trapz <- function(xx, yy) {
    if (length(xx) < 2) return(NA_real_)
    sum((yy[-1] + yy[-length(yy)]) * diff(xx) / 2)
  }
  
  total <- trapz(f2, p2)
  out <- list(total_0p5_45 = total)
  
  for (nm in names(bands)) {
    lo <- bands[[nm]][1]; hi <- bands[[nm]][2]
    idx <- which(f2 >= lo & f2 < hi)
    bp <- if (length(idx) >= 2) trapz(f2[idx], p2[idx]) else NA_real_
    out[[paste0("bp_", nm)]] <- bp
    out[[paste0("rp_", nm)]] <- if (is.finite(total) && total > 0 && is.finite(bp)) bp / total else NA_real_
  }
  
  if (length(f2) >= 2 && is.finite(total) && total > 0) {
    cum <- rep(0, length(f2))
    for (i in 2:length(f2)) {
      cum[i] <- cum[i-1] + (p2[i] + p2[i-1]) * (f2[i] - f2[i-1]) / 2
    }
    target <- 0.95 * cum[length(cum)]
    out$sef95_0p5_45 <- f2[which(cum >= target)[1] %||% length(f2)]
  } else {
    out$sef95_0p5_45 <- NA_real_
  }
  
  out
}

extract_features_one <- function(edf_path, max_channels = 2, from_min = 0, dur_min = 8) {
  chans <- choose_eeg_channels(edf_path, max_n = max_channels)
  if (!length(chans)) stop("No usable EEG channels detected (channels.tsv missing or edfReader unable to parse signals).")
  
  from_s <- as.numeric(from_min) * 60
  till_s <- from_s + as.numeric(dur_min) * 60
  
  sigs <- read_edf_window(edf_path, chans, from_s, till_s)
  if (!length(sigs)) stop("Could not read EDF signals for requested window.")
  
  per_channel <- list()
  for (nm in names(sigs)) {
    x  <- sigs[[nm]]$x
    fs <- sigs[[nm]]$fs
    x <- x[is.finite(x)]
    if (length(x) < 256) next
    
    bp <- band_powers(x, fs)
    
    per_channel[[nm]] <- c(
      mean = mean(x),
      sd = sd(x),
      median = median(x),
      mad = mad(x, constant = 1.4826),
      iqr = IQR(x),
      rms = rms1(x),
      ll  = line_length(x),
      zcr = zero_cross_rate(x),
      skew = skewness1(x),
      kurt = kurtosis1(x),
      unlist(bp)
    )
  }
  
  if (!length(per_channel)) stop("Signals read, but not enough finite samples to extract features.")
  
  feats <- list()
  for (nm in names(per_channel)) {
    v <- per_channel[[nm]]
    names(v) <- paste0("ch_", make.names(nm), "__", names(v))
    feats <- c(feats, as.list(v))
  }
  
  mat <- do.call(rbind, lapply(per_channel, as.numeric))
  colnames(mat) <- names(per_channel[[1]])
  pooled <- colMeans(mat, na.rm = TRUE)
  names(pooled) <- paste0("meanCh__", names(pooled))
  feats <- c(feats, as.list(pooled))
  
  out <- suppressWarnings(as.numeric(unlist(feats)))
  names(out) <- names(feats)
  out
}

apply_plotly_theme <- function(p, xlab=NULL, ylab=NULL){
  if (!is.null(xlab)) p <- layout(p, xaxis = list(title = list(text = xlab)))
  if (!is.null(ylab)) p <- layout(p, yaxis = list(title = list(text = ylab)))
  p <- plotly::layout(
    p,
    font = list(family = "Ramabhadra, Arial, sans-serif", size = 12),
    paper_bgcolor = page_bg, plot_bgcolor  = page_bg,
    margin = list(l = 54, r = 36, t = 36, b = 54),
    xaxis = list(gridcolor = "#E8E8E8", zeroline = FALSE, fixedrange = TRUE,
                 automargin = FALSE, ticks = "outside", ticklen = 8, title = list(standoff = 28)),
    yaxis = list(gridcolor = "#E8E8E8", zeroline = FALSE, fixedrange = TRUE,
                 automargin = FALSE, ticks = "outside", ticklen = 8, title = list(standoff = 28)),
    hovermode = "x unified",
    hoverlabel = list(align = "left", bgcolor = "#fff", bordercolor = "#000", font = list(size = 14)),
    dragmode = FALSE
  )
  plotly::config(
    p, scrollZoom = FALSE, doubleClick = FALSE,
    modeBarButtonsToRemove = list(
      "zoom2d","pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d",
      "autoScale2d","resetScale2d","toggleSpikelines","toImage"
    ),
    displaylogo = FALSE, displayModeBar = TRUE
  )
}

# =============================================================================
# UI (draggable tiles)
# =============================================================================
ui <- fluidPage(
  tags$head(
    tags$title("PSG Analysis Tool"),
    tags$style(HTML("@import url('https://fonts.googleapis.com/css2?family=Ramabhadra:wght@400;700;900&display=swap');")),
    tags$style(HTML(sprintf("
      html { scroll-behavior: smooth; }
      body { margin: 0; padding: 0; background: %s; color: #313131; font-family: 'Ramabhadra', Arial, sans-serif; line-height: 1.5; }
      .shiny-notification { display: none !important; }

      .qtb-actions { display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.75rem; }
      .qtb-title { font-size: 1.6rem; font-weight: 900; color: %s; margin: 0; }
      .qtb-sub { color: slategray; margin: 0.25rem 0 0 0; }
      .qtb-btn {
        display:inline-flex; align-items:center; gap:.35rem;
        font-family:'Ramabhadra', Arial, sans-serif; font-size:0.75em; font-weight:600; text-transform:uppercase;
        padding:0.3em 0.75em;
        color: slategray !important; border:1px solid slategray; border-radius:0.25rem; background:transparent;
        cursor:pointer; transition: all 0.2s ease-in-out;
      }
      .qtb-btn:hover, .qtb-btn:focus { color:%s !important; border-color:%s; outline:none; }

      #notebook { width: 100%%; max-width: 1100px; margin: 20px auto; background: %s; border-radius: 8px; }
      .grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 12px; padding: 12px; }
      .tile.full { grid-column: 1 / -1; }
      @media(max-width:980px){ .grid { grid-template-columns: 1fr; } }

      .tile { height: %dpx; background: %s; border: 1px solid #CFCFCF; border-radius: .25rem; overflow: hidden; display: flex; flex-direction: column; }
      #tile_controls, #tile_status { height: auto !important; }
      #tile_controls .tile-body, #tile_status .tile-body { flex: 0 0 auto !important; overflow: visible !important; padding: 12px; }

      .tile-hdr { display:flex; align-items:center; gap:8px; padding:8px 10px; border-bottom:1px solid %s; background:#FBFBFB; }
      .tile-title { font-weight:700; color:#222; font-size:14px; letter-spacing:.2px; text-transform:uppercase; }
      .grab { cursor: grab; width: 22px; height: 22px; border-radius: .25rem; display: inline-grid; place-items: center; background: #FFFFFF; border: 1px solid #CFCFCF; }
      .grab:active{ cursor: grabbing; }
      .grab svg{ width: 16px; height: 16px; stroke-width: 2; color: #333; display: block; }

      .tile-body { flex:1; padding:10px; overflow:auto; min-height:0; }
      .tile-body .js-plotly-plot, .tile-body .plotly { width: 100%% !important; height: 100%% !important; }

      .ui-sortable-placeholder { visibility: visible !important; border: 1px dashed #9E9E9E; background: %s !important; border-radius: .25rem; min-height: %dpx; }

      label { font-size: 12px; color: #444; font-weight: 700; font-family: 'Ramabhadra', Arial, sans-serif; }
      .form-control, .selectize-input, input[type='number'], input[type='text'] { background: #fff; color: #222; border: 1px solid #CFCFCF; border-radius: .25rem; font-family: 'Ramabhadra', Arial, sans-serif; }
      .dataTables_length, .dataTables_filter, .dataTables_info { display: none !important; }

      /* Status tile */
      .status-row { display:flex; align-items:flex-start; gap:10px; padding:8px 10px; border:1px solid #E0E0E0; border-radius:6px; background:#FFFFFF; }
      .badge { font-size: 11px; font-weight: 800; letter-spacing:.2px; padding: 2px 8px; border-radius: 999px; border:1px solid transparent; text-transform:uppercase; }
      .b-ok{ color:#1f7a1f; border-color:#1f7a1f22; background:#1f7a1f11; }
      .b-warn{ color:#8a5a00; border-color:#8a5a0022; background:#8a5a0011; }
      .b-err{ color:#8a1f1f; border-color:#8a1f1f22; background:#8a1f1f11; }
      .b-run{ color:%s; border-color:%s22; background:%s11; }

      .status-msg{ font-weight:800; color:#222; margin:0; }
      .status-detail{ color: slategray; margin: 2px 0 0 0; font-size: 12px; }
      .status-time{ color: slategray; font-size: 11px; white-space:nowrap; }

      .prog-wrap{ width:100%%; background:#EFEDE6; border:1px solid #DAD8D2; border-radius: 6px; overflow:hidden; height:12px; }
      .prog-bar{ height:12px; background:%s; width:0%%; transition: width 120ms linear; }
      .prog-meta{ display:flex; justify-content:space-between; align-items:center; margin: 6px 0 10px 0; color: slategray; font-size: 12px; }

      /* Plotly notifier off */
      .plotly-notifier { display: none !important; }
    ",
                            page_bg, brand_color, brand_color, brand_color, nb_bg, TILE_H, nb_bg, menu_border,
                            page_bg, TILE_H, brand_color, brand_color, brand_color, brand_color
    ))),
    tags$head(
      tags$style(HTML("@import url('https://fonts.googleapis.com/css2?family=Ramabhadra:wght@400;700;900&display=swap');")),
      tags$style(HTML("
    :root{
      --page-bg: #F2F0E9;
      --nb-bg:   #FAF8F1;
      --border:  #E0E0E0;
      --brand:   #751F2C;
      --accent:  #A04559; /* slider + upload progress */
    }

    body{
      background: var(--page-bg);
      color:#313131;
      font-family: 'Ramabhadra', Arial, sans-serif;
      margin:0;
    }

    /* ===== Quarto-style header (matches HeadLines) ===== */
    .quarto-title-banner{
      background:#F5F5F5;
      border-bottom:1px solid var(--border);
      padding: 1.45rem;
      box-sizing: border-box;
      width: 100%;
    }
    .qtb-inner{
      max-width:1100px;
      margin:0 auto;
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    .qtb-title{
      font-size: 1.18em;
      font-weight: 900;
      color: var(--brand);
      margin:0;
      line-height: 1.1;
    }
    .qtb-actions{
      display:flex;
      flex-wrap:wrap;
      gap: .5rem;
      margin-top: .25rem;
    }

    /* Header buttons */
    .qtb-btn{
      display:inline-flex;
      align-items:center;
      gap:.35rem;
      font-family:'Ramabhadra', Arial, sans-serif;
      font-size: .75em;
      font-weight: 600;
      text-transform: uppercase;

      padding: 0.3em 0.75em;
      color: slategray !important;
      border: 1px solid slategray;
      border-radius: .25rem;
      background: transparent !important;
      cursor: pointer;
      transition: all 0.2s ease-in-out;
      box-shadow: none !important;
    }
    .qtb-btn:hover, .qtb-btn:focus{
      color: var(--brand) !important;
      border-color: var(--brand) !important;
      outline: none !important;
    }

    .btn.btn-default.btn-file:hover{
      color: var(--brand) !important;
      border-color: var(--brand) !important;
    }

    /* Upload progress bar color */
    .progress-bar{ background-color: var(--accent) !important; }
    .progress{ background: #E8E8E8 !important; }

    /* Slider (ionRangeSlider) colors */
    .irs-min, .irs-max, .irs-grid-text{ font-family:'Ramabhadra', Arial, sans-serif; }
    .irs--shiny .irs-from,
    .irs--shiny .irs-to,
    .irs--shiny .irs-single{
      color:#fff;
      text-shadow:none;
      font-family:'Ramabhadra', Arial, sans-serif;
      padding: 1px 3px;
      background-color: var(--brand);
      border-radius: 3px;
      font-size: 11px;
      line-height: 1.333;
    }
    .irs--shiny .irs-bar{
      top: 25px;
      height: 8px;
      border-top: 1px solid var(--accent);
      border-bottom: 1px solid var(--accent);
      background: var(--accent);
    }
  "))
    )
  ),
  
  # Header
  div(
    class = "quarto-title-banner",
    div(
      class = "qtb-inner",
      h1(class = "qtb-title", "PSG ML Workbench"),
      div(
        class = "qtb-actions",
        actionButton("index_bids",  "INDEX BIDS",     class = "qtb-btn"),
        actionButton("build_feats", "BUILD FEATURES", class = "qtb-btn"),
        actionButton("train_model", "TRAIN XGBOOST",  class = "qtb-btn"),
        actionButton("predict_all", "PREDICT",        class = "qtb-btn"),
        downloadButton("dl_preds",  "DOWNLOAD CSV",   class = "qtb-btn")
      )
    )
  ),
  
  # Notebook container
  div(
    id = "notebook",
    jqui_sortable(
      div(
        id = "grid", class = "grid",
        
        # Controls (full)
        div(
          id = "tile_controls", class = "tile full",
          div(class = "tile-hdr", span(class = "grab", MOVE_SVG), span(class = "tile-title", "Data & Controls")),
          div(
            class = "tile-body",
            fluidRow(
              column(
                6,
                radioButtons(
                  "data_mode",
                  "Dataset source",
                  choices = c("Upload BIDS ZIP" = "zip", "Server path (Shiny Server)" = "path"),
                  selected = "zip",
                  inline = TRUE
                ),
                conditionalPanel(
                  condition = "input.data_mode == 'zip'",
                  fileInput("bids_zip", "BIDS dataset (.zip)", accept = c(".zip")),
                  tags$div(style="color:slategray;font-size:12px;margin-top:6px;",
                           "Local is fine. On shinyapps.io, EDF ZIPs often exceed upload limits; use Shiny Server path mode if needed.")
                ),
                conditionalPanel(
                  condition = "input.data_mode == 'path'",
                  textInput("bids_path", "Server BIDS root path", value = "", placeholder = "/srv/shiny-server/data/bids_root"),
                  tags$div(style="color:slategray;font-size:12px;margin-top:6px;",
                           "Path must exist on the machine running the app.")
                ),
                fileInput("participants_tsv", "participants.tsv (optional override)", accept = c(".tsv", ".txt")),
                uiOutput("target_ui")
              ),
              column(
                6,
                fluidRow(
                  column(4, numericInput("window_min", "Window duration (minutes)", value = 8, min = 1, max = 30, step = 1)),
                  column(4, numericInput("max_channels", "Max EEG channels", value = 2, min = 1, max = 4, step = 1)),
                  column(4, numericInput("seed", "Seed", value = 42, min = 1, step = 1))
                ),
                fluidRow(
                  column(6, sliderInput("test_prop", "Test split (%)", min = 15, max = 40, value = 25, step = 5)),
                  column(6, selectInput("xgb_mode", "Tuning mode", choices = c("Fast" = "fast", "Better" = "better"), selected = "better"))
                ),
                tags$div(style="color:slategray;font-size:12px;margin-top:6px;",
                         "Better = small randomized search + early stopping. Fast = strong defaults + early stopping.")
              )
            )
          )
        ),
        
        # Status (full)
        div(
          id = "tile_status", class = "tile full",
          div(class = "tile-hdr", span(class = "grab", MOVE_SVG), span(class = "tile-title", "Status & Progress")),
          div(class = "tile-body",
              uiOutput("progress_ui"),
              tags$div(style="margin-top:8px;", uiOutput("status_ui"))
          )
        ),
        
        # Index table (full)
        div(
          id = "tile_index", class = "tile full",
          div(class = "tile-hdr", span(class = "grab", MOVE_SVG), span(class = "tile-title", "Indexed EDF files")),
          div(class = "tile-body", DTOutput("tbl_index"))
        ),
        
        # Summary tile (half)
        div(
          id = "tile_summary", class = "tile",
          div(class = "tile-hdr", span(class = "grab", MOVE_SVG), span(class = "tile-title", "Run Summary")),
          div(class = "tile-body", uiOutput("summary_ui"))
        ),
        
        # ROC tile (half)
        div(
          id = "tile_roc", class = "tile",
          div(class = "tile-hdr", span(class = "grab", MOVE_SVG), span(class = "tile-title", "ROC (binary only)")),
          div(class = "tile-body", plotlyOutput("plot_roc", height = "100%"))
        ),
        
        # Hist tile (half)
        div(
          id = "tile_hist", class = "tile",
          div(class = "tile-hdr", span(class = "grab", MOVE_SVG), span(class = "tile-title", "Probability histogram")),
          div(class = "tile-body", plotlyOutput("plot_hist", height = "100%"))
        ),
        
        # Importance (full)
        div(
          id = "tile_imp", class = "tile full",
          div(class = "tile-hdr", span(class = "grab", MOVE_SVG), span(class = "tile-title", "XGBoost feature importance")),
          div(class = "tile-body", plotlyOutput("plot_imp", height = "100%"))
        ),
        
        # Predictions (full)
        div(
          id = "tile_preds", class = "tile full",
          div(class = "tile-hdr", span(class = "grab", MOVE_SVG), span(class = "tile-title", "Predictions")),
          div(class = "tile-body", DTOutput("tbl_preds"))
        )
      ),
      options = list(
        tolerance = "pointer",
        placeholder = "ui-sortable-placeholder",
        handle = ".tile-hdr"
      )
    )
  ),
  
  tags$script(HTML(sprintf('const DEFAULT_STATE = %s;', toJSON(DEFAULT_STATE, auto_unbox = TRUE)))),
  tags$script(HTML('
    function collectOrder(){ return Array.from(document.querySelectorAll("#grid > .tile")).map(el=>el.id); }
    function applyOrder(order){ const grid=document.getElementById("grid"); (order||[]).forEach(id=>{ const el=document.getElementById(id); if(el) grid.appendChild(el); }); }

    Shiny.addCustomMessageHandler("setProgressBar", function(payload){
      var bar = document.querySelector(".prog-bar");
      if(!bar) return;
      var p = Math.max(0, Math.min(1, payload.p || 0));
      bar.style.width = (p*100).toFixed(0) + "%";
    });

    document.addEventListener("DOMContentLoaded", function(){
      const saved = localStorage.getItem("psg_grid_state");
      const start = saved ? JSON.parse(saved) : DEFAULT_STATE;
      applyOrder(start.order);

      $(document).on("sortstop", "#grid", function(){
        const order=collectOrder();
        localStorage.setItem("psg_grid_state", JSON.stringify({order}));
        Shiny.setInputValue("grid_order", order, {priority:"event"});
      });
    });
  '))
)

# =============================================================================
# Server
# =============================================================================
server <- function(input, output, session){
  
  # ---- Rich status + progress state ----
  status_log <- reactiveVal(data.frame(
    time = character(), level = character(), msg = character(), detail = character(),
    stringsAsFactors = FALSE
  ))
  prog_state <- reactiveVal(list(p = 0, label = "Idle", right = ""))
  
  push_status <- function(level = c("ok","warn","err","run"), msg, detail = "") {
    level <- match.arg(level)
    df <- status_log()
    df <- rbind(df, data.frame(
      time = timestamp_est(), level = level, msg = msg, detail = detail,
      stringsAsFactors = FALSE
    ))
    if (nrow(df) > 60) df <- tail(df, 60)
    status_log(df)
  }
  
  set_prog <- function(p, label, right = "") {
    prog_state(list(p = p, label = label, right = right))
    session$sendCustomMessage("setProgressBar", list(p = p))
  }
  
  reset_prog <- function(label = "Idle") set_prog(0, label, "")
  
  # initial system check
  observeEvent(TRUE, {
    push_status("ok", "App ready", "Upload a BIDS ZIP or point to a server path, then click INDEX BIDS.")
    pk <- c(
      edfReader = has_pkg("edfReader"),
      xgboost   = has_pkg("xgboost"),
      pROC      = has_pkg("pROC")
    )
    missing <- names(pk)[!unlist(pk)]
    if (length(missing)) {
      push_status("warn", "Optional packages missing",
                  paste0("Install for full functionality: ", paste(missing, collapse = ", "),
                         ". (Local: install.packages(c('edfReader','xgboost','pROC')))"))
    }
  }, once = TRUE)
  
  # ---- Dataset root resolution (ZIP or server path) ----
  bids_root <- reactiveVal(NULL)
  extracted_dir <- reactiveVal(NULL)
  
  observeEvent(list(input$data_mode, input$bids_zip, input$bids_path), {
    if (identical(input$data_mode, "path")) {
      p <- trimws(input$bids_path %||% "")
      if (nzchar(p) && dir.exists(p)) {
        bids_root(p)
        push_status("ok", "Using server dataset path", p)
      } else {
        bids_root(NULL)
        if (nzchar(p)) push_status("err", "Server path not found", p)
      }
      return()
    }
    
    if (is.null(input$bids_zip)) {
      bids_root(NULL)
      return()
    }
    
    tmp <- tempfile("bids_", tmpdir = tempdir())
    dir.create(tmp, recursive = TRUE, showWarnings = FALSE)
    extracted_dir(tmp)
    
    push_status("run", "Unzipping dataset", basename(input$bids_zip$name))
    set_prog(0.05, "Unzipping", input$bids_zip$name)
    
    tryCatch({
      utils::unzip(input$bids_zip$datapath, exdir = tmp)
      root <- detect_bids_root(tmp)
      bids_root(root)
      push_status("ok", "Dataset ready", paste0("Root: ", root))
      reset_prog("Idle")
    }, error = function(e){
      bids_root(NULL)
      push_status("err", "Failed to unzip dataset", e$message)
      reset_prog("Idle")
    })
  }, ignoreInit = TRUE)
  
  # ---- Participants + target ----
  participants_df <- reactiveVal(NULL)
  target_col <- reactiveVal("group")
  
  observeEvent(list(bids_root(), input$participants_tsv), {
    df <- NULL
    
    if (!is.null(input$participants_tsv)) {
      df <- safe_read_tsv(input$participants_tsv$datapath)
      if (!is.null(df)) push_status("ok", "Loaded participants.tsv override", input$participants_tsv$name)
    } else if (!is.null(bids_root())) {
      auto <- file.path(bids_root(), "participants.tsv")
      if (file.exists(auto)) {
        df <- safe_read_tsv(auto)
        if (!is.null(df)) push_status("ok", "Loaded participants.tsv from dataset root", auto)
      }
    }
    
    if (!is.null(df) && nrow(df)) {
      if (!("participant_id" %in% names(df))) {
        idc <- names(df)[tolower(names(df)) %in% c("participant_id", "subject", "sub", "id")][1] %||% NA_character_
        if (is.character(idc) && nzchar(idc)) names(df)[names(df) == idc] <- "participant_id"
      }
      participants_df(df)
      if ("group" %in% names(df)) target_col("group")
      else {
        cand <- setdiff(names(df), "participant_id")
        target_col(cand[1] %||% "group")
      }
    } else {
      participants_df(NULL)
    }
  }, ignoreInit = TRUE)
  
  output$target_ui <- renderUI({
    df <- participants_df()
    if (is.null(df)) {
      return(tags$div(style="color:slategray;font-size:12px;",
                      "participants.tsv not loaded yet. Training needs labels (e.g., group)."))
    }
    selectInput("target", "Target label column", choices = names(df), selected = target_col() %||% "group")
  })
  
  observeEvent(input$target, {
    if (!is.null(input$target) && nzchar(input$target)) target_col(input$target)
  }, ignoreInit = TRUE)
  
  # ---- Index EDF ----
  files_index <- reactiveVal(NULL)
  
  output$tbl_index <- renderDT({
    datatable(
      data.frame(INFO = "Click INDEX BIDS after providing a dataset."),
      rownames = FALSE,
      options = list(dom = "t", paging = FALSE, scrollX = TRUE),
      selection = "none"
    )
  })
  
  observeEvent(input$index_bids, {
    root <- bids_root()
    if (is.null(root) || !dir.exists(root)) {
      push_status("err", "No dataset root available",
                  "Upload a ZIP or set a server path first.")
      return()
    }
    
    push_status("run", "Indexing EDF files", root)
    set_prog(0.1, "Indexing EDFs", "")
    
    idx <- NULL
    tryCatch({
      idx <- index_bids_edf(root)
    }, error = function(e){
      push_status("err", "Indexing failed", e$message)
    })
    
    if (is.null(idx) || !nrow(idx)) {
      push_status("err", "No EDF files found",
                  "Expected EDFs under sub-*/eeg/*_eeg.edf (or any .edf).")
      reset_prog("Idle")
      return()
    }
    
    idx <- idx[!is.na(idx$participant_id) & nzchar(idx$participant_id), , drop = FALSE]
    if (!nrow(idx)) {
      push_status("err", "No participant_id detected in EDF paths",
                  "Expected BIDS-style sub-XX in folder or filename.")
      reset_prog("Idle")
      return()
    }
    
    idx <- idx[order(idx$participant_id, idx$edf_path), , drop = FALSE]
    idx <- idx[!duplicated(idx$participant_id), , drop = FALSE]
    
    part <- participants_df()
    if (!is.null(part) && "participant_id" %in% names(part)) {
      idx <- merge(idx, part, by = "participant_id", all.x = TRUE)
    }
    
    files_index(idx)
    
    output$tbl_index <- renderDT({
      datatable(idx, rownames = FALSE, options = list(dom = "t", paging = FALSE, scrollX = TRUE))
    })
    
    push_status("ok", "Index complete",
                paste0("Participants indexed: ", nrow(idx), " (one EDF per participant)"))
    reset_prog("Idle")
  }, ignoreInit = TRUE)
  
  # ---- Feature extraction ----
  features_df <- reactiveVal(NULL)
  
  observeEvent(input$build_feats, {
    idx <- files_index()
    if (is.null(idx) || !nrow(idx)) {
      push_status("err", "No index available", "Run INDEX BIDS first.")
      return()
    }
    if (!has_pkg("edfReader")) {
      push_status("err", "Missing package: edfReader", "Install: install.packages('edfReader')")
      return()
    }
    
    maxc <- as.integer(input$max_channels %||% 2)
    durm <- as.numeric(input$window_min %||% 8)
    
    push_status("run", "Building features", paste0("Window=", durm, " min • Max EEG channels=", maxc))
    set_prog(0.0, "Feature extraction", "0%")
    
    out_rows <- list()
    n <- nrow(idx)
    
    tryCatch({
      for (i in seq_len(n)) {
        set_prog(i / n, "Feature extraction", paste0(round(100 * i / n), "%"))
        p_id <- idx$participant_id[i]
        edf  <- idx$edf_path[i]
        
        fv <- tryCatch(
          extract_features_one(edf_path = edf, max_channels = maxc, from_min = 0, dur_min = durm),
          error = function(e) {
            push_status("warn", paste0("Skipped ", p_id), e$message)
            NULL
          }
        )
        if (is.null(fv)) next
        
        row <- data.frame(
          participant_id = p_id,
          edf_path = edf,
          stringsAsFactors = FALSE
        )
        
        # carry metadata from index (including label columns)
        for (nm in setdiff(names(idx), c("edf_path", "filename"))) {
          if (!nm %in% names(row)) row[[nm]] <- idx[[nm]][i]
        }
        
        for (nm in names(fv)) row[[nm]] <- fv[[nm]]
        out_rows[[length(out_rows) + 1]] <- row
      }
      
      feat <- if (length(out_rows)) do.call(rbind, out_rows) else NULL
      if (is.null(feat) || !nrow(feat)) {
        push_status("err", "Feature extraction produced zero rows",
                    "Check that EDFs are readable and channels.tsv marks EEG channels as type=EEG.")
        reset_prog("Idle")
        return()
      }
      
      features_df(feat)
      push_status("ok", "Features ready",
                  paste0("Rows: ", nrow(feat), " • Numeric features: ",
                         sum(vapply(feat, is.numeric, logical(1)))))
      reset_prog("Idle")
    }, error = function(e){
      push_status("err", "Feature extraction crashed", e$message)
      reset_prog("Idle")
    })
  }, ignoreInit = TRUE)
  
  # ---- XGBoost training ----
  model_obj <- reactiveVal(NULL)
  preds_df <- reactiveVal(NULL)
  
  train_xgb <- function(df, label_col, seed = 42, test_prop = 0.25, mode = c("fast","better")) {
    mode <- match.arg(mode)
    if (!has_pkg("xgboost")) stop("Missing package: xgboost (install.packages('xgboost'))")
    
    d <- df[!is.na(df[[label_col]]) & nzchar(as.character(df[[label_col]])), , drop = FALSE]
    if (nrow(d) < 12) stop("Not enough labeled participants to train (need ~12+).")
    
    y_fac <- as.factor(d[[label_col]])
    lv <- levels(y_fac)
    k <- length(lv)
    if (k < 2) stop("Target must have at least 2 classes.")
    
    drop_cols <- intersect(names(d), c("edf_path", "filename", label_col))
    d2 <- d[, setdiff(names(d), drop_cols), drop = FALSE]
    keep_num <- vapply(d2, is.numeric, logical(1))
    X <- d2[, keep_num, drop = FALSE]
    if (ncol(X) < 10) stop("Too few numeric features for xgboost training.")
    
    # median impute
    for (j in seq_len(ncol(X))) {
      v <- X[[j]]
      if (anyNA(v)) {
        med <- median(v, na.rm = TRUE)
        v[is.na(v)] <- med
        X[[j]] <- v
      }
    }
    
    set.seed(seed)
    idx_all <- seq_len(nrow(X))
    
    # stratified-ish test split
    idx_test <- c()
    for (lvv in lv) {
      ids <- idx_all[y_fac == lvv]
      n_te <- max(1, floor(length(ids) * test_prop))
      idx_test <- c(idx_test, sample(ids, n_te))
    }
    idx_test <- sort(unique(idx_test))
    idx_train <- setdiff(idx_all, idx_test)
    
    X_train <- as.matrix(X[idx_train, , drop = FALSE])
    X_test  <- as.matrix(X[idx_test,  , drop = FALSE])
    
    if (k == 2) {
      y_train <- as.integer(y_fac[idx_train] == lv[2])
      y_test  <- as.integer(y_fac[idx_test]  == lv[2])
      
      dtrain <- xgboost::xgb.DMatrix(X_train, label = y_train)
      dtest  <- xgboost::xgb.DMatrix(X_test,  label = y_test)
      
      base_params <- list(
        objective = "binary:logistic",
        eval_metric = "auc",
        booster = "gbtree",
        eta = 0.05,
        max_depth = 4,
        min_child_weight = 1,
        subsample = 0.85,
        colsample_bytree = 0.85,
        gamma = 0,
        lambda = 1,
        alpha = 0
      )
      
      candidates <- list(base_params)
      if (mode == "better") {
        for (i in 1:8) {
          candidates[[length(candidates)+1]] <- modifyList(base_params, list(
            eta = sample(c(0.03, 0.05, 0.08), 1),
            max_depth = sample(3:6, 1),
            min_child_weight = sample(c(1,2,4), 1),
            subsample = sample(c(0.75, 0.85, 0.95), 1),
            colsample_bytree = sample(c(0.7, 0.85, 1.0), 1),
            gamma = sample(c(0, 0.5, 1), 1)
          ))
        }
      }
      
      best <- NULL
      best_auc <- -Inf
      
      for (pp in candidates) {
        fit <- xgboost::xgb.train(
          params = pp,
          data = dtrain,
          nrounds = 2000,
          watchlist = list(train = dtrain, eval = dtest),
          early_stopping_rounds = 40,
          verbose = 0
        )
        pr <- predict(fit, dtest)
        auc <- NA_real_
        if (has_pkg("pROC")) {
          roc <- pROC::roc(y_test, pr, quiet = TRUE)
          auc <- as.numeric(pROC::auc(roc))
        } else {
          auc <- mean((pr >= 0.5) == (y_test == 1))
        }
        if (is.finite(auc) && auc > best_auc) {
          best_auc <- auc
          best <- fit
        }
      }
      
      pr_test <- predict(best, dtest)
      pred_test <- ifelse(pr_test >= 0.5, lv[2], lv[1])
      
      list(
        fit = best,
        levels = lv,
        k = k,
        X = X,
        y_fac = y_fac,
        idx_train = idx_train,
        idx_test = idx_test,
        prob_test = pr_test,
        pred_test = factor(pred_test, levels = lv),
        metric = list(auc = best_auc)
      )
      
    } else {
      # multiclass softprob
      y_int <- as.integer(y_fac) - 1L
      y_train <- y_int[idx_train]
      y_test  <- y_int[idx_test]
      
      dtrain <- xgboost::xgb.DMatrix(X_train, label = y_train)
      dtest  <- xgboost::xgb.DMatrix(X_test,  label = y_test)
      
      params <- list(
        objective = "multi:softprob",
        num_class = k,
        eval_metric = "mlogloss",
        eta = 0.06,
        max_depth = 5,
        subsample = 0.85,
        colsample_bytree = 0.85
      )
      
      fit <- xgboost::xgb.train(
        params = params,
        data = dtrain,
        nrounds = 1500,
        watchlist = list(train = dtrain, eval = dtest),
        early_stopping_rounds = 40,
        verbose = 0
      )
      
      pr <- matrix(predict(fit, dtest), ncol = k, byrow = TRUE)
      pred <- apply(pr, 1, which.max) - 1L
      acc <- mean(pred == y_test)
      
      list(
        fit = fit,
        levels = lv,
        k = k,
        X = X,
        y_fac = y_fac,
        idx_train = idx_train,
        idx_test = idx_test,
        prob_test = pr,
        pred_test = factor(lv[pred + 1L], levels = lv),
        metric = list(accuracy = acc)
      )
    }
  }
  
  observeEvent(input$train_model, {
    feat <- features_df()
    if (is.null(feat) || !nrow(feat)) {
      push_status("err", "No features available", "Run BUILD FEATURES first.")
      return()
    }
    
    lab <- target_col() %||% "group"
    if (!(lab %in% names(feat))) {
      push_status("err", "Label column not found in features", paste0("Missing: ", lab, " • Load participants.tsv and re-index."))
      return()
    }
    
    push_status("run", "Training xgboost", paste0("Target=", lab, " • Mode=", input$xgb_mode %||% "better"))
    set_prog(0.05, "Training", "Starting")
    
    tryCatch({
      set_prog(0.15, "Training", "Splitting + preprocessing")
      fit <- train_xgb(
        df = feat,
        label_col = lab,
        seed = as.integer(input$seed %||% 42),
        test_prop = as.numeric(input$test_prop %||% 25) / 100,
        mode = input$xgb_mode %||% "better"
      )
      model_obj(fit)
      
      if (fit$k == 2) {
        push_status("ok", "Model trained", paste0("Binary AUC = ", sprintf("%.3f", fit$metric$auc)))
      } else {
        push_status("ok", "Model trained", paste0("Multiclass accuracy = ", sprintf("%.3f", fit$metric$accuracy)))
      }
      reset_prog("Idle")
    }, error = function(e){
      push_status("err", "Training failed", e$message)
      reset_prog("Idle")
    })
  }, ignoreInit = TRUE)
  
  # ---- Predict ----
  observeEvent(input$predict_all, {
    feat <- features_df()
    fit <- model_obj()
    
    if (is.null(feat) || !nrow(feat)) {
      push_status("err", "No features available", "Run BUILD FEATURES first.")
      return()
    }
    if (is.null(fit)) {
      push_status("err", "No trained model available", "Run TRAIN XGBOOST first.")
      return()
    }
    
    push_status("run", "Generating predictions", "")
    set_prog(0.1, "Predicting", "Preparing matrix")
    
    tryCatch({
      X <- fit$X
      for (j in seq_len(ncol(X))) {
        v <- X[, j]
        if (anyNA(v)) {
          med <- median(v, na.rm = TRUE)
          v[is.na(v)] <- med
          X[, j] <- v
        }
      }
      
      dmat <- xgboost::xgb.DMatrix(as.matrix(X))
      
      if (fit$k == 2) {
        set_prog(0.6, "Predicting", "Scoring")
        p <- predict(fit$fit, dmat)
        pred <- ifelse(p >= 0.5, fit$levels[2], fit$levels[1])
        out <- feat
        out$.pred_prob <- p
        out$.pred_class <- factor(pred, levels = fit$levels)
        preds_df(out)
      } else {
        set_prog(0.6, "Predicting", "Scoring")
        pr <- matrix(predict(fit$fit, dmat), ncol = fit$k, byrow = TRUE)
        pred <- apply(pr, 1, which.max)
        out <- feat
        out$.pred_class <- factor(fit$levels[pred], levels = fit$levels)
        preds_df(out)
      }
      
      push_status("ok", "Predictions ready", paste0("Rows: ", nrow(preds_df())))
      reset_prog("Idle")
    }, error = function(e){
      push_status("err", "Prediction failed", e$message)
      reset_prog("Idle")
    })
  }, ignoreInit = TRUE)
  
  # ---- Status UI ----
  output$progress_ui <- renderUI({
    ps <- prog_state()
    pct <- round(100 * (ps$p %||% 0))
    div(
      div(class="prog-meta",
          span(strong(ps$label %||% "Idle")),
          span(ps$right %||% paste0(pct, "%"))
      ),
      div(class="prog-wrap", div(class="prog-bar")),
      tags$div(style="margin-top:10px;color:slategray;font-size:12px;",
               "If a button appears to do nothing, check this tile: it will tell you exactly what’s missing or what failed.")
    )
  })
  
  output$status_ui <- renderUI({
    df <- status_log()
    if (is.null(df) || !nrow(df)) return(NULL)
    
    # newest first
    df <- df[rev(seq_len(nrow(df))), , drop = FALSE]
    
    badge_class <- function(level) {
      switch(level,
             ok = "badge b-ok",
             warn = "badge b-warn",
             err = "badge b-err",
             run = "badge b-run",
             "badge b-warn")
    }
    badge_label <- function(level) {
      switch(level,
             ok = "OK",
             warn = "WARN",
             err = "ERROR",
             run = "RUNNING",
             toupper(level))
    }
    
    items <- lapply(seq_len(nrow(df)), function(i){
      div(
        class = "status-row",
        span(class = badge_class(df$level[i]), badge_label(df$level[i])),
        div(
          style="flex:1;",
          p(class="status-msg", df$msg[i]),
          if (nzchar(df$detail[i] %||% "")) p(class="status-detail", df$detail[i])
        ),
        span(class="status-time", df$time[i])
      )
    })
    
    div(style="display:flex;flex-direction:column;gap:8px;", items)
  })
  
  # ---- Summary tile ----
  output$summary_ui <- renderUI({
    idx <- files_index()
    feat <- features_df()
    fit <- model_obj()
    pr <- preds_df()
    
    rows <- list(
      tags$div(style="font-weight:900;color:#222;text-transform:uppercase;font-size:12px;", "Snapshot"),
      tags$div(style="color:slategray;font-size:12px;margin-top:6px;",
               paste0("EDF indexed: ", if (is.null(idx)) "—" else nrow(idx))),
      tags$div(style="color:slategray;font-size:12px;",
               paste0("Feature rows: ", if (is.null(feat)) "—" else nrow(feat))),
      tags$div(style="color:slategray;font-size:12px;",
               paste0("Model: ", if (is.null(fit)) "—" else paste0("xgboost (", if (fit$k==2) "binary" else "multiclass", ")"))),
      tags$div(style="color:slategray;font-size:12px;",
               paste0("Predictions: ", if (is.null(pr)) "—" else nrow(pr)))
    )
    
    if (!is.null(fit)) {
      if (fit$k == 2) {
        rows <- c(rows, tags$div(style="margin-top:10px;color:#222;font-weight:900;font-size:12px;text-transform:uppercase;",
                                 paste0("Test AUC: ", sprintf("%.3f", fit$metric$auc))))
      } else {
        rows <- c(rows, tags$div(style="margin-top:10px;color:#222;font-weight:900;font-size:12px;text-transform:uppercase;",
                                 paste0("Test accuracy: ", sprintf("%.3f", fit$metric$accuracy))))
      }
    }
    
    div(rows)
  })
  
  # ---- ROC / Hist / Importance ----
  output$plot_roc <- renderPlotly({
    fit <- model_obj()
    validate(need(!is.null(fit), "Train xgboost to see ROC."))
    validate(need(fit$k == 2, "ROC shown for binary targets only."))
    validate(need(has_pkg("pROC"), "Install pROC to plot ROC: install.packages('pROC')"))
    
    y_test <- as.integer(fit$y_fac[fit$idx_test] == fit$levels[2])
    pr <- fit$prob_test
    roc <- pROC::roc(y_test, pr, quiet = TRUE)
    auc <- as.numeric(pROC::auc(roc))
    
    df <- data.frame(fpr = 1 - roc$specificities, tpr = roc$sensitivities)
    gg <- ggplot(df, aes(fpr, tpr)) +
      geom_line(linewidth = 1, color = brand_color) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "slategray") +
      labs(x = "FALSE POSITIVE RATE",, y = "TRUE POSITIVE RATE",
           title = paste0("ROC (AUC = ", sprintf("%.3f", auc), ")")) +
      theme_minimal(base_family = "Ramabhadra")
    
    apply_plotly_theme(ggplotly(gg, dynamicTicks = TRUE))
  })
  
  output$plot_hist <- renderPlotly({
    pr <- preds_df()
    validate(need(!is.null(pr) && nrow(pr) > 0, "Run PREDICT to see histogram."))
    validate(need(".pred_prob" %in% names(pr), "Histogram shown for binary targets only."))
    
    x <- pr$.pred_prob
    x <- x[is.finite(x)]
    
    gg <- ggplot(data.frame(p = x), aes(p)) +
      geom_histogram(bins = 30, fill = red_pal$pale, alpha = 0.95) +
      labs(x = "P(positive class)", y = "COUNT") +
      theme_minimal(base_family = "Ramabhadra")
    
    apply_plotly_theme(ggplotly(gg, dynamicTicks = TRUE))
  })
  
  output$plot_imp <- renderPlotly({
    fit <- model_obj()
    validate(need(!is.null(fit), "Train xgboost to see feature importance."))
    validate(need(has_pkg("xgboost"), "xgboost not available."))
    
    imp <- xgboost::xgb.importance(model = fit$fit)
    validate(need(!is.null(imp) && nrow(imp) > 0, "No importance available."))
    
    imp <- imp[order(imp$Gain, decreasing = TRUE), ]
    imp <- head(imp, 20)
    
    gg <- ggplot(imp, aes(x = reorder(Feature, Gain), y = Gain)) +
      geom_col(fill = brand_color) +
      coord_flip() +
      labs(x = NULL, y = "GAIN", title = "Top 20 features") +
      theme_minimal(base_family = "Ramabhadra")
    
    apply_plotly_theme(ggplotly(gg, dynamicTicks = TRUE))
  })
  
  # ---- Predictions table + download ----
  output$tbl_preds <- renderDT({
    pr <- preds_df()
    if (is.null(pr)) {
      return(datatable(
        data.frame(INFO = "INDEX BIDS → BUILD FEATURES → TRAIN XGBOOST → PREDICT"),
        rownames = FALSE, options = list(dom = "t", paging = FALSE, scrollX = TRUE),
        selection = "none"
      ))
    }
    key <- c("participant_id", target_col() %||% "group", ".pred_class", ".pred_prob", "filename")
    key <- key[key %in% names(pr)]
    rest <- setdiff(names(pr), key)
    show <- pr[, c(key, rest), drop = FALSE]
    
    datatable(show, rownames = FALSE, options = list(dom = "tip", pageLength = 25, scrollX = TRUE))
  })
  
  output$dl_preds <- downloadHandler(
    filename = function() paste0("psg_predictions_", format(Sys.time(), "%Y-%m-%d_%H%M%S"), ".csv"),
    content = function(file) {
      pr <- preds_df()
      validate(need(!is.null(pr) && nrow(pr) > 0, "No predictions to export."))
      write.csv(pr, file, row.names = FALSE)
    }
  )
}

shinyApp(ui, server)
