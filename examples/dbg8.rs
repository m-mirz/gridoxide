use gridoxide::rao::{crac_json, evaluate_with, Network, Resolution};
fn main(){
    let net = gridoxide::ucte::read("tests/data/ucte/TestCase12Nodes.uct").unwrap();
    let (crac,_) = crac_json::read("tests/data/rao/features/SL_ep4us2_4MR_A.json").unwrap();
    let res = Resolution::new(&crac, &net.branch_ids);
    let tap: i32 = std::env::args().nth(1).unwrap().parse().unwrap();
    let angle = crac.range_actions.iter().find_map(|r| r.kind.angle_at(tap)).unwrap();
    let mut xf = net.transformers.clone();
    let ratio = xf[0].tap.norm();
    xf[0].tap = num_complex::Complex::from_polar(ratio, (-angle).to_radians());
    let nw = Network { buses:&net.buses, lines:&net.lines, transformers:&xf,
                       branch_ids:&net.branch_ids, base_mva:net.base_mva };
    println!("tap {tap} ({angle:+.4} deg)");
    for p in evaluate_with(&crac,&nw,&res,&[]).perimeters {
        for c in &p.cnecs {
            println!("  {:<48} flow {:9.2} limit {:9.2} margin {:8.2}",
                crac.flow_cnecs[c.cnec].id, c.flow_mw, c.limit_mw, c.margin_mw);
        }
    }
}
