parallel --shuf --bar --timeout 3060 --jobs 4 -u ./target/release/pcp-rec-str {1} {2} 30 ::: 0 1 2 3 4 6 9 12 15 17 21 22 23 24 28 29 32 34 35 36 37 38 39 40 44 45 ::: "false" "true"     