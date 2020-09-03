require(ggplot2)

# ===== Coefficient table processing =====

# Add 'Term' and 'Value' columns to biglm coefficient table to replace concatenated 'TermValue' column
add_label <- function(fit) {
  
  # Add a category label column next to category values and coefficients
  terms <- attr(fit$terms,"term.labels")
  map <- data.frame(index=0:length(terms),Term=c("(Intercept)", terms))
  m <- data.frame(summary(fit)$mat)
  m$index <- fit$assign
  m <- merge(m, map, "index", drop=T)
  m$index <- NULL
  # Extract category values from concatenated label,value-strings
  if (attr(fit$terms,"intercept"))
    terms <- c("(Intercept)", terms)
  m$Term <- factor(m$Term, terms)
  m$Value <- fit$names
  m$Value <- apply(m, 1, function(x) substring(x["Value"], nchar(x["Term"])+1, nchar(x["Value"])))
  return(m)
}

# Expand coefficient table of a single term to include all given values
add_term_levels <- function(term, all_levels, fr, reference=T) {
  if (!is.null(all_levels)) {
    values <- data.frame(Value=all_levels, temp=1:length(all_levels))
    fr <- merge(x=fr, y=values, all.y=TRUE)
    fr <- fr[order(fr$temp), ]
    fr$temp <- NULL
  }  
  fr$Term[is.na(fr$Term)] <- term
  if (reference)
    fr$Coef[is.na(fr$Coef)] <- 0.00
  return(fr)
}

# Iterate over coefficient table to add all possible values to each term
add_levels <- function(df, coef) {
  all_levels <- lapply(df, levels)
  temp <- list()
  all_fr <- split(coef, coef$Term)
  for (term in names(all_fr)) {
    temp[[term]] <- add_term_levels(term, all_levels[[term]], all_fr[[term]])
  }
  coef.new <- do.call(rbind, temp)
  rownames(coef.new) <- 1:nrow(coef.new)
  return (coef.new)
}



plot_oddsratio <- function(fr) {
  term <- fr$Term[[1]]
  #fn <- paste('odds', term, '.png', sep='')
  #png(fn, width = 70+20*nrow(fr), height=300, res=72)
  fr$Value = factor(fr$Value, levels=unique(fr$Value))
  fr$Coef <- exp(fr$Coef)
  fr$X.95. <- exp(fr$X.95.)
  fr$CI. <- exp(fr$CI.)
  pd <- position_dodge(0.1)
  p <- ggplot(fr, aes(x=Value, y=Coef)) + 
    geom_hline(yintercept = 1, linetype='dashed') + 
    geom_errorbar(aes(ymin=X.95., ymax=CI.), colour="black", width=.1, position=pd) +
    geom_point(size=3, shape=21, fill="white", position=pd) + # 21 is filled circle
    xlab(term) +
    ylab("Relative Odds") +
    ggtitle("Unemployment Prevalence") +
    theme_bw() +
    scale_y_continuous(trans='log2') + 
    theme(axis.text.x = element_text(angle = 90, hjust = 1), plot.title = element_text(hjust = 0.5))
  print(p)
  #dev.off()
}

