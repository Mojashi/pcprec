# 0 to 200
parallel --shuf --bar --timeout 12 --jobs 2 -u ./target/release/pcp-rec-str {1} > jobs_res/{1}.out ::: $(seq 0 200)