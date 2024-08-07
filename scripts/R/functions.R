
## functions for CI using Boostrap ------------------------- 
result_performance_ci  <-
  function(score, set, obj, formu, n_boot = 100) {
    
    result_ibs <- result_c <- c()
    for(i in 1:n_boot){
      nrows <- nrow(set)
      index <- sample(1:nrows, nrows, replace = TRUE)
      set_tmp <- set[index,]
      score_tmp <- score[index]
      
      # c-index
      Surv_test_tmp <- Surv(set_tmp$label_time, set_tmp$label_status)
      AUC_c <- concordancefit(Surv_test_tmp, -score_tmp)
      result_c <- append(result_c, AUC_c$concordance)
      
      # ibs
      suppressMessages({
      brier_tmp <- pec::pec( object = obj, data = set_tmp, 
                             formula = formu, 
                             splitMethod = "none")
      })
      ibs_tmp <- crps(brier_tmp, start = min(set_tmp$label_time), times = max(set_tmp$label_time))
      result_ibs <- append(result_ibs, ibs_tmp[2]) 
    }
    
    result_all <- data.frame(result_c, result_ibs)
    est_mean <- colMeans(result_all)
    se <- apply(result_all, 2, sd)
    up <- sapply(result_all, function(x){quantile(x,probs = c(0.975))})
    down <- sapply(result_all, function(x){quantile(x,probs = c(0.025))})
    p_value <- 2 * pt( abs(est_mean/se), df = Inf, lower.tail = FALSE )
    
    result_final <- paste0(round(est_mean,4)," (",round(down,4),"-",round(up,4),")")
    result_p <- paste0(p_value, " (",round(se, 4),")")
    
    cat('----------------------------------------------', '\n',
        "C_index: ", result_final[1], '\n', "p value (se):",  result_p[1], '\n',
        '----------------------------------------------', '\n',
        "Integral Brier score: ", result_final[2], '\n', "p value (se):", result_p[2],'\n',
        '----------------------------------------------', '\n' )
    
    table_result <- cbind(est_mean, se, down, up)
    row_name <- c("Cindex", "IBS")
    col_name <- c("Mean", "SE", "lower_q", "upper_q")
    dimnames(table_result) <- list(row_name, col_name)
    
    return(table_result)
    
  }


## functions for Autoscore_survival ------------------------- 
assign_score_mod <- function(df, score_table) {
  for (i in setdiff(names(df), c("label", "label_time", "label_status"))) {
    score_table_tmp <-
      score_table[grepl(i, names(score_table))]
    df[, i] <- as.character(df[, i])
    
    if (length(score_table_tmp)>3) {
      for (j in 1:length(names(score_table_tmp))) {
        pattern <- gsub(i, "", names(score_table_tmp)[j])
        df[, i][df[, i] %in% pattern] <- score_table_tmp[j]
      } }
    
    df[, i] <- as.factor(df[, i])
    #df[, i] <- as.numeric(df[, i])   
  }
  
  return(df)
}

get_brier_score <- function(test_set, variables){
  surv_formula <- as.formula(paste0("Surv(label_time,label_status)~",
                                    paste0(variables,collapse="+")))
  coxph_mod <- coxph(surv_formula,data=test_set,x=TRUE,y=TRUE)

  pec_mod <- pec(coxph_mod, formula = surv_formula, data = test_set)
  
  pec_brier <- crps(pec_mod, start = min(test_set$label_time), times = max(test_set$label_time))
  return(pec_brier[2])

}

get_brier_score_ci <- function(test_set, scoring_table, variables, cut_vec, n_boot = 100) {
  
  test_set <- test_set[,c(variables, "label_time","label_status")]
  result_all <- c()
  
  test_set_trans <- AutoScore::transform_df_fixed(test_set, cut_vec = cut_vec)
  test_set_score <- assign_score_mod(test_set_trans, scoring_table)
  
  print("Running bootstrapped samples")
  for(i in 1:n_boot){
    nrows <- nrow(test_set)
    index <- sample(1:nrows,nrows,replace = TRUE)
    test_set_tmp <- test_set_score[index, ]
    
    surv_formula <- as.formula(paste0("Surv(label_time,label_status)~",
                                      paste0(variables,collapse="+")))
    coxph_mod <- coxph(Surv(label_time,label_status)~., data=test_set_tmp, x=TRUE, y=TRUE)
    while(is.na(sum(coxph_mod$coefficients))==TRUE){
      index <- sample(1:nrows,nrows,replace = TRUE)
      test_set_tmp <- test_set_score[index,]
      coxph_mod <- coxph(surv_formula, data=test_set_tmp, x=TRUE, y=TRUE)
    }
    
    #calculate IBS
    suppressMessages({
    pec_mod <- pec(coxph_mod, formula = Surv(label_time, label_status)~1, data = test_set_tmp)
    })
    pec_brier <- crps(pec_mod, start = min(test_set_tmp$label_time), times = max(test_set_tmp$label_time))
    test_set_tmp_brier <- pec_brier[2]
    
    result_all <- append(result_all, test_set_tmp_brier)
  }
  
  est_mean <- mean(result_all)
  up <- quantile(result_all ,probs = c(0.975))
  down <- quantile(result_all ,probs = c(0.025))
  result_final<-paste0(round(est_mean,4)," (",round(down,4),"-",round(up,4),")")
  return(result_final)
  
}

plot_importance <- function(ranking){
  df = data.frame(Imp = ranking, Index = factor(names(ranking), levels = rev(names(ranking))))
  p <- ggplot(data = df, mapping = aes_string(y = "Index", x = "Imp")) +
    geom_rect(
      aes(xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf),
      fill = NA,
      color = "black",
      size =0.5
    )+
    geom_bar(stat = "identity", fill = "#2b8cbe", width = 0.7) +
    scale_y_discrete(expand = expansion(mult = 0, add = 1)) +
    labs(x = "Variable Importance", y = "", title = "Importance Ranking") +
    theme(
      plot.title = element_text(hjust = 0.5, vjust = 0.3),
      panel.grid.major = element_line(colour = "gray"),
      #panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "white"),
      axis.line = element_line(colour = "black"),
      axis.text = element_text(size = 10,colour = "black"),
      axis.text.x = element_text(angle = 0, vjust = 0, hjust = 0))
  if(nrow(df) > 25){
    p <- p + theme(axis.text.y = element_text(size = 13.5 - 1.5 * (floor(nrow(df)/21))-1))
  }

  print(p)
}

AutoScore_impute <- function(train_set, validation_set = NULL){
  n_train <- nrow(train_set)
  df <- rbind(train_set, validation_set)
  for (i in 1:ncol(df)){

    if (names(df)[i] == "label" & sum(is.na(df[, i])) > 0){
      stop("There are missing values in the outcome: label! Please fix it and try again")
    }

    if (names(df)[i] == "label_time" & sum(is.na(df[, i])) > 0){
      stop("There are missing values in the outcome: label_time!Please fix it and try again")
    }

    if (names(df)[i] == "label_status" & sum(is.na(df[, i])) > 0){
      stop("There are missing values in the outcome:lable_status!Please fix it and try again")
    }

    var = df[1:n_train, i]
    if (is.factor(df[, i]) & sum(is.na(df[, i])) > 0){
      df[is.na(df[, i]), i] <- getmode(var)
    } else if (is.numeric(var) & sum(is.na(var)) > 0) {
      df[is.na(df[, i]), i] <- median(var, na.rm=T)
    }
  }

  if (is.null(validation_set)){
    return(df)
  } else {
    train_set <- df[1:n_train, ]
    validation_set <- df[(n_train+1):nrow(df), ]
    return(list("train" = train_set, "validation" = validation_set))
  }
}

AutoScore_rank_Survival_fast <- function(train_set, ntree, mytre, nodesize, nsplit) {
  
  train_set <- AutoScore_impute(train_set)
  model <-
    rfsrc(
      Surv(label_time, label_status) ~ .,
      train_set,
      mytre = mytre,
      nodesize = nodesize,
      nsplit = nsplit,
      ntree = ntree,   
      save.memory = TRUE,
      perf.type = "none", 
      do.trace = T,
      seed = 30 
    )
  
  # estimate variable importance
  importance <- vimp(model)
  importance_a <-  sort( importance$importance, decreasing = T)
  cat("The ranking based on variable importance was shown below for each variable: \n")
  print(importance_a)
  
  #plot_importance(importance_a)
  
  return(importance_a)
  
} 


generate_sample <- function(nn, mean, lower_limit, upper_limit) {
  confidence_width <- upper_limit - lower_limit
  standard_deviation <- confidence_width / (2 * 1.96) 
  sample_data <- rnorm(nn, mean = mean, sd = standard_deviation)
  return(sample_data)
}