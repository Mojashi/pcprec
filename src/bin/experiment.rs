use itertools::Itertools;
use pcp_rec_str::pcp::PCP;
use pcp_rec_str::union_pdr::union_pdr;
use std::io::BufRead;
use std::process::exit;
fn parse_file(f: &str) -> Vec<(String, PCP)> {
    let mut ret = vec![];
    let f = std::fs::File::open(f).unwrap();
    let lines = std::io::BufReader::new(f).lines().collect_vec();
    for result in lines.iter() {
        let record = result.as_ref().unwrap();
        let pcp = PCP::parse_pcp_string(&record.as_str());
        ret.push((record.to_string(), pcp));
    }
    ret
}

fn main() {
    let args = std::env::args().collect::<Vec<String>>();
    let rev = args.get(1).unwrap().parse::<bool>().unwrap();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();

    let mut pcp = PCP::parse_pcp_string(input.trim());
    let revpcp = pcp.reverse_pcp();
    if rev {
        pcp = revpcp;
    }
    println!("{:?}", pcp);
    let res = union_pdr(pcp);
    if res {
        exit(0);
    } else {
        exit(1);
    }
}
