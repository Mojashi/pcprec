use itertools::Itertools;
use std::io::BufRead;

use pcp_rec_str::pcp::PCP;
use pcp_rec_str::union_pdr2::union_pdr;
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
    let pcp = PCP::parse_pcp_string("PCP(Vector(Tile(1111,1), Tile(1,1001), Tile(0,11)))");
    
    let pcp = PCP::parse_pcp_string("PCP(Vector(Tile(1110,1), Tile(01,11), Tile(1,011)))");
    //let pcp = PCP::parse_pcp_string("PCP(Vector(Tile(1111,1), Tile(1,1001), Tile(0,11)))");
    let pcp = PCP::parse_pcp_string("PCP(Vector(Tile(1110,1), Tile(1,10), Tile(0,1110)))");//30
    let pcp = PCP::parse_pcp_string("PCP(Vector(Tile(10,0), Tile(0,001), Tile(001,1)))");
    let pcp = PCP::parse_pcp_string("PCP(Vector(Tile(1111,1), Tile(1,1110), Tile(0,1111)))");//20

    let pcp = PCP::parse_pcp_string("PCP(Vector(Tile(1111,11), Tile(10,111), Tile(1,1110)))").reverse_pcp();//0

    // let args = std::env::args().collect::<Vec<String>>();
    // let idx = args.get(1).unwrap().parse::<usize>().unwrap();
    // let rev = args.get(2).unwrap().parse::<bool>().unwrap();
    // //let idx = 31;
    // let instances = parse_file("a.csv");
    // let (raw,pcp) = instances.get(idx).unwrap().clone();
    // let pcp = if rev { pcp.reverse_pcp() } else { pcp };
    // println!("{idx} {rev} {:?}", pcp);

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let pcp = PCP::parse_pcp_string(input.trim());

    // let pcp = PCP::parse_pcp_string("PCP(Vector(Tile(1101,1), Tile(0110,11), Tile(1,110)))");//hard
    // let pcp = PCP::parse_pcp_string("PCP(Vector(Tile(11101,1), Tile(10,1110), Tile(1,10)))");//hard
    // let pcp = PCP::parse_pcp_string("PCP(Vector(Tile(100,1)Tile(0,100)Tile(1,00))))");//solvable
    // let pcp = PCP::parse_pcp_string("PCP(Vector(Tile(1111,1), Tile(10,11), Tile(1,1110)))");//20
    // let pcp = PCP::parse_pcp_string("PCP(Vector(Tile(11011,1), Tile(0110,11), Tile(1,110)))");//hard
    // let pcp = PCP::parse_pcp_string("PCP(Vector(Tile(110,1), Tile(1,0), Tile(0,110)))").re;//hard

    union_pdr(pcp);
    //println!("finishedpcp {idx} {rev}");
}
