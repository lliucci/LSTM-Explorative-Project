library(imputeTS)
library(tidyverse)

theme_set(theme_bw())

Data = read_csv("Data/Cleaned_Data.csv")

P33 = Data %>% 
    filter(X.stn == "P33") %>%
    select(date, Depth)

P33$Depth_Int = na_kalman(P33$Depth)

interpolation = P33 %>%
    ggplot(aes(x = date)) +
    geom_line(aes(y = Depth_Int), color = 'red') +
    geom_line(aes(y = Depth), color = 'black') +
    labs(x = "Date", y = "Depth")

ggsave(interpolation, 
    filename = "Deliverables/Written Component/Figures/Interpolation.png",
    scale = 1,
    height = 1000,
    width = 2000,
    units = 'px')

interpolation_80_00 = P33 %>%
    filter(date >= "1980-01-01" & date <= "2000-10-1") %>%
    ggplot(aes(x = date)) +
    geom_line(aes(y = Depth_Int), color = 'red') +
    geom_line(aes(y = Depth), color = 'black') +
    labs(x = "Date", y = "Depth")

ggsave(interpolation_80_00, 
    filename = "Deliverables/Written Component/Figures/Interpolation_60_20.png",
    scale = 1,
    height = 1000,
    width = 2000,
    units = 'px')

P33 = P33 %>%
    select(date, Depth_Int) %>%
    rename(Depth = Depth_Int)

write_csv(P33, "Data/P33.csv")
