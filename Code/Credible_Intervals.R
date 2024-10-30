library(tidyverse)
library(bayestestR)
theme_set(theme_bw())

Pred_Intervals = read_csv("Data/Pred_Intervals.csv")
P33 = read_csv("Data/P33.csv") %>% filter(date >= "1980-01-01") %>% select(Depth)
Test = P33[14205:14235,]

Pred_Intervals_t = Pred_Intervals %>%
    as.matrix() %>%
    t() %>%
    as_tibble()

Plotting = tibble(hdi(Pred_Intervals_t)) %>% 
    bind_cols(Test) %>%
    mutate(Index = seq(1, 31, 1))

Credible_Intervals = Plotting %>%
    ggplot() +
    geom_line(aes(x = Index, y = Depth, color = 'blue'), lwd = 2) +
    geom_ribbon(aes(x = Index, ymin = CI_low, ymax = CI_high, color = 'red'), alpha = 0.25, lwd = 2, lty = 2) +
    labs(x = "Days Out of Training Data",
        y = "Depth") +
    scale_colour_manual(name = 'Source',
        guide = 'legend',
        values =c('blue'='blue','red'='red'), labels = c('Observed','95% CI'))

ggsave(Credible_Intervals,
    filename = "Deliverables/Written Component/Figures/Credible_Intervals.png",
    scale = 2,
    width = 1600,
    height = 800,
    units = 'px')

